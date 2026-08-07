import { describe, expect, it } from "vitest";

import {
  describeFields,
  initialValues,
  jsonErrors,
  missingRequired,
  outOfRange,
  overrideCount,
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

/**
 * A closed set has to render as a closed set.
 *
 * These are the exact shapes from `EfficientAdConfig` and `PreprocessingConfig`, and every
 * one of them used to reach the wrong control: `Literal` became a free text box, because
 * `type: "string"` was tested before `enum`; `StrEnum` became a JSON textarea, because
 * `$ref` was not in the type at all and nothing resolved `$defs`.
 */
const ENUM_SCHEMA: OptionsSchema = {
  $defs: {
    ColorMode: { enum: ["rgb", "grayscale"], title: "ColorMode", type: "string" },
    Resample: {
      description: "Named rather than numeric, so a stored config still reads as a decision.",
      enum: ["nearest", "bilinear", "bicubic", "lanczos"],
      title: "Resample",
      type: "string",
    },
    Window: { properties: { size: { type: "integer" } }, title: "Window", type: "object" },
  },
  properties: {
    model_size: {
      default: "small",
      description: "PDN capacity.",
      enum: ["small", "medium"],
      title: "Model Size",
      type: "string",
    },
    color: { $ref: "#/$defs/ColorMode", default: "rgb", description: "Colour to feed the model." },
    resample: { $ref: "#/$defs/Resample", default: "bilinear" },
    fallback_mode: {
      anyOf: [{ $ref: "#/$defs/ColorMode" }, { type: "null" }],
      default: null,
      title: "Fallback Mode",
    },
    window: { $ref: "#/$defs/Window", default: null },
    missing: { $ref: "#/$defs/NotThere", default: null },
    level: { default: 2, enum: [1, 2, 3], title: "Level", type: "integer" },
  },
};

const enumFields = describeFields(ENUM_SCHEMA);
const enumField = (name: string) => enumFields.find((entry) => entry.name === name)!;

describe("a closed set of values", () => {
  it("reads `enum` before `type`, so a Literal is not a text box", () => {
    // The whole bug in one assertion: `Literal["small","medium"]` carries both, and
    // `type: "string"` used to win.
    expect(enumField("model_size").kind).toBe("choice-inline");
    expect(enumField("model_size").options).toEqual([
      { value: "small", label: "Small" },
      { value: "medium", label: "Medium" },
    ]);
  });

  it("follows a $ref into $defs, so a StrEnum is not a JSON textarea", () => {
    expect(enumField("color").kind).toBe("choice-inline");
    expect(enumField("color").options.map((option) => option.value)).toEqual([
      "rgb",
      "grayscale",
    ]);
  });

  it("shows a few options at once and puts many behind a picker", () => {
    expect(enumField("color").kind).toBe("choice-inline");
    expect(enumField("resample").kind).toBe("choice");
  });

  it("resolves a $ref hidden inside the `X | None` encoding", () => {
    expect(enumField("fallback_mode").kind).toBe("choice-inline");
  });

  it("labels the field, not the enum class it points at", () => {
    // Dereferencing pulls in `title: "ColorMode"`; using it would label the control with
    // the name of a Python type the operator has never heard of.
    expect(enumField("color").label).toBe("Color");
    expect(enumField("resample").label).toBe("Resample");
  });

  it("prefers the field's own description to the enum type's docstring", () => {
    expect(enumField("color").description).toBe("Colour to feed the model.");
    // `resample` states nothing itself, so the type's docstring is better than nothing.
    expect(enumField("resample").description).toMatch(/Named rather than numeric/);
  });

  it("carries the default through the $ref, since pydantic writes it as a sibling", () => {
    expect(enumField("color").fallback).toBe("rgb");
    expect(enumField("resample").fallback).toBe("bilinear");
  });

  it("still falls back to a JSON textarea for a $ref that is not a closed set", () => {
    // The escape hatch has to survive: an operator can express what the model accepts even
    // where the form has nothing prettier to offer.
    expect(enumField("window").kind).toBe("json");
  });

  it("falls back rather than throwing when a $ref points at nothing", () => {
    expect(enumField("missing").kind).toBe("json");
  });
});

describe("sending a choice", () => {
  it("sends nothing for a choice nobody touched", () => {
    // The segment matching the default is highlighted, but highlighting is not choosing:
    // the default stays defined in Python alone.
    const options = toOptions(enumFields, initialValues(enumFields));

    expect(options).not.toHaveProperty("model_size");
    expect(options).not.toHaveProperty("color");
  });

  it("sends the value once it is chosen", () => {
    const options = toOptions(enumFields, {
      ...initialValues(enumFields),
      model_size: "medium",
    });

    expect(options["model_size"]).toBe("medium");
  });

  it("sends a numeric enum as a number, not as the string the control handed back", () => {
    const options = toOptions(enumFields, { ...initialValues(enumFields), level: "3" });

    expect(options["level"]).toBe(3);
  });
});

/**
 * The bounds were in the schema the whole time. Dropping them meant an out-of-range value
 * was reported as a 422 from the backend, half a screen away from the field that caused it.
 */
const BOUNDED_SCHEMA: OptionsSchema = {
  properties: {
    max_steps: { type: "integer", default: 4000, minimum: 10, maximum: 200000 },
    learning_rate: { type: "number", default: 0.0001, exclusiveMinimum: 0, maximum: 1 },
    seed: { type: "integer", default: 0 },
  },
};

const bounded = describeFields(BOUNDED_SCHEMA);
const boundedField = (name: string) => bounded.find((entry) => entry.name === name)!;

describe("numeric bounds", () => {
  it("carries the schema's own limits onto the control", () => {
    expect(boundedField("max_steps").min).toBe(10);
    expect(boundedField("max_steps").max).toBe(200000);
  });

  it("steps an integer by one and frees a float entirely", () => {
    // A number input steps by 1 unless told otherwise, which marks a learning rate of
    // 0.0001 invalid and refuses to submit the form.
    expect(boundedField("max_steps").step).toBe(1);
    expect(boundedField("learning_rate").step).toBe("any");
  });

  it("states an exclusive bound as exclusive, which min/max cannot", () => {
    expect(boundedField("learning_rate").range).toBe("> 0, ≤ 1");
    expect(boundedField("max_steps").range).toBe("≥ 10, ≤ 200000");
    expect(boundedField("seed").range).toBeNull();
  });

  it("names a value outside the range, so the button can block on it", () => {
    const values = { ...initialValues(bounded), max_steps: "9" };

    expect(outOfRange(bounded, values)).toEqual(["Max steps"]);
  });

  it("says nothing about an untouched field or one inside the range", () => {
    expect(outOfRange(bounded, initialValues(bounded))).toEqual([]);
    expect(outOfRange(bounded, { ...initialValues(bounded), max_steps: "4000" })).toEqual([]);
  });
});

describe("overrideCount", () => {
  it("counts nothing when the form is untouched, whatever the defaults are", () => {
    // Including the booleans, which start at their default and are always sent.
    expect(overrideCount(fields, initialValues(fields))).toBe(0);
  });

  it("counts each field the operator actually changed", () => {
    const values = { ...initialValues(fields), csv_path: "a.csv", defect_type_from_dir: false };

    expect(overrideCount(fields, values)).toBe(2);
  });
});
