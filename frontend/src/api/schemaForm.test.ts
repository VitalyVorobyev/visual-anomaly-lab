import { describe, expect, it } from "vitest";

import {
  describeFields,
  initialValues,
  jsonErrors,
  missingRequired,
  toOptions,
  type OptionsSchema,
} from "./schemaForm";

/** The shapes pydantic actually emits, copied from a live `/api/import/adapters`. */
const SCHEMA: OptionsSchema = {
  required: ["csv_path"],
  properties: {
    csv_path: { type: "string", title: "Csv Path", description: "The table." },
    image_column: { type: "string", title: "Image Column", default: "image" },
    label_column: {
      anyOf: [{ type: "string" }, { type: "null" }],
      title: "Label Column",
      default: "label",
    },
    filter_column: {
      anyOf: [{ type: "string" }, { type: "null" }],
      title: "Filter Column",
      default: null,
    },
    normal_values: {
      type: "array",
      items: { type: "string" },
      title: "Normal Values",
      default: ["normal", "good"],
    },
    defect_type_from_dir: { type: "boolean", title: "Defect Type From Dir", default: true },
    max_rows: { type: "integer", title: "Max Rows", default: 10 },
    channel_aliases: {
      type: "object",
      additionalProperties: { type: "string" },
      title: "Channel Aliases",
    },
  },
};

const fields = describeFields(SCHEMA);
const field = (name: string) => fields.find((entry) => entry.name === name)!;

describe("describeFields", () => {
  it("picks a control from the schema node's type", () => {
    expect(field("csv_path").kind).toBe("text");
    expect(field("normal_values").kind).toBe("string-list");
    expect(field("defect_type_from_dir").kind).toBe("boolean");
    expect(field("max_rows").kind).toBe("number");
    expect(field("channel_aliases").kind).toBe("json");
  });

  it("sees through the `str | None` encoding to the type that carries meaning", () => {
    // Without unwrapping, every optional string would fall through to a JSON textarea.
    expect(field("label_column").kind).toBe("text");
    expect(field("filter_column").kind).toBe("text");
  });

  it("humanises the field name rather than echoing pydantic's generated title", () => {
    expect(field("csv_path").label).toBe("Csv path");
    expect(field("defect_type_from_dir").label).toBe("Defect type from dir");
  });

  it("keeps a title a human actually chose", () => {
    const described = describeFields({
      properties: { root: { type: "string", title: "Where the images live" } },
    });

    expect(described[0]?.label).toBe("Where the images live");
  });

  it("marks what the schema requires", () => {
    expect(field("csv_path").required).toBe(true);
    expect(field("image_column").required).toBe(false);
  });

  it("shows the backend's default as a placeholder", () => {
    expect(field("image_column").placeholder).toBe("image");
    expect(field("normal_values").placeholder).toBe("normal, good");
  });
});

describe("what an operator has to decide", () => {
  it("keeps a field with no useful default in front of them", () => {
    // `filter_column` defaults to null: the adapter cannot filter until you say on what.
    expect(field("filter_column").advanced).toBe(false);
  });

  it("keeps a required field in front of them", () => {
    expect(field("csv_path").advanced).toBe(false);
  });

  it("folds away a field the adapter already answers", () => {
    expect(field("image_column").advanced).toBe(true);
    expect(field("normal_values").advanced).toBe(true);
    expect(field("defect_type_from_dir").advanced).toBe(true);
  });

  it("treats an empty list default as something still to supply", () => {
    // This is the rule that puts `normal_dirs` and `defect_dirs` at the top of the
    // `folder_classes` form, which is the whole simplified-selection ask.
    const [dirs] = describeFields({
      properties: { normal_dirs: { type: "array", items: { type: "string" }, default: [] } },
    });

    expect(dirs?.advanced).toBe(false);
  });
});

describe("initialValues", () => {
  it("starts text and list controls empty even where a default exists", () => {
    const values = initialValues(fields);

    // A pre-filled box would claim the default as the operator's choice, and would send
    // it back on every scan — pinning a default that later changes.
    expect(values["image_column"]).toBe("");
    expect(values["normal_values"]).toBe("");
  });

  it("starts a checkbox at its default, because it has no empty state", () => {
    expect(initialValues(fields)["defect_type_from_dir"]).toBe(true);
  });
});

describe("toOptions", () => {
  it("leaves out anything untouched, so the backend's default stays the only one", () => {
    const options = toOptions(fields, initialValues(fields));

    expect(Object.keys(options)).toEqual(["defect_type_from_dir"]);
  });

  it("splits a comma-separated list and drops the blanks", () => {
    const options = toOptions(fields, {
      ...initialValues(fields),
      normal_values: " good , , ok ",
    });

    expect(options["normal_values"]).toEqual(["good", "ok"]);
  });

  it("sends numbers as numbers", () => {
    const options = toOptions(fields, { ...initialValues(fields), max_rows: "42" });

    expect(options["max_rows"]).toBe(42);
  });

  it("always sends a checkbox, in both positions", () => {
    expect(toOptions(fields, { ...initialValues(fields), defect_type_from_dir: false })).toEqual({
      defect_type_from_dir: false,
    });
  });

  it("parses a JSON object field", () => {
    const options = toOptions(fields, {
      ...initialValues(fields),
      channel_aliases: '{"illumb": "bright"}',
    });

    expect(options["channel_aliases"]).toEqual({ illumb: "bright" });
  });

  it("leaves out JSON that does not parse rather than sending a string", () => {
    const options = toOptions(fields, { ...initialValues(fields), channel_aliases: "{oops" });

    expect(options).not.toHaveProperty("channel_aliases");
  });

  it("trims a plain string", () => {
    expect(toOptions(fields, { ...initialValues(fields), csv_path: "  a.csv  " })).toMatchObject({
      csv_path: "a.csv",
    });
  });
});

describe("validation", () => {
  it("names the JSON field that does not parse", () => {
    expect(jsonErrors(fields, { ...initialValues(fields), channel_aliases: "{oops" })).toEqual([
      "Channel aliases",
    ]);
  });

  it("says nothing about an empty JSON field", () => {
    expect(jsonErrors(fields, initialValues(fields))).toEqual([]);
  });

  it("names a required field the operator has not filled in", () => {
    expect(missingRequired(fields, initialValues(fields))).toEqual(["Csv path"]);
  });

  it("is satisfied once it is filled in", () => {
    const values = { ...initialValues(fields), csv_path: "split.csv" };

    expect(missingRequired(fields, values)).toEqual([]);
  });

  it("treats whitespace as unfilled", () => {
    const values = { ...initialValues(fields), csv_path: "   " };

    expect(missingRequired(fields, values)).toEqual(["Csv path"]);
  });
});

describe("an adapter with no options at all", () => {
  it("produces no fields and no options", () => {
    const none = describeFields({});

    expect(none).toEqual([]);
    expect(toOptions(none, {})).toEqual({});
  });
});
