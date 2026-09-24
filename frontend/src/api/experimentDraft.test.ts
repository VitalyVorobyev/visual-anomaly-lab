import { describe, expect, it } from "vitest";

import { clearDraft, draftKey, readDraft, writeDraft, type ExperimentDraft } from "./experimentDraft";

function memory(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key) => store.get(key) ?? null,
    key: (index) => [...store.keys()][index] ?? null,
    removeItem: (key) => void store.delete(key),
    setItem: (key, value) => void store.set(key, value),
  };
}

const DRAFT: ExperimentDraft = {
  name: "kept",
  datasetId: 7,
  splitId: 3,
  regionProfileId: 11,
  methodKey: "pixel_reference",
  configValues: { quantile: "0.99" },
  preprocessingValues: {},
  evaluationValues: { rule: "f1" },
  channels: ["top"],
};

describe("the create-experiment draft", () => {
  it("round-trips through storage", () => {
    const storage = memory();
    writeDraft("k", DRAFT, storage);
    expect(readDraft("k", storage)).toEqual(DRAFT);
    clearDraft("k", storage);
    expect(readDraft("k", storage)).toBeNull();
  });

  it("keeps the dataset-scoped form and the cross-dataset one apart", () => {
    expect(draftKey(7)).not.toBe(draftKey(undefined));
  });

  it("drops anything malformed instead of trusting it", () => {
    const storage = memory();
    storage.setItem("k", "{not json");
    expect(readDraft("k", storage)).toBeNull();
    storage.setItem("k", JSON.stringify({ name: 3, datasetId: "7", channels: ["a", 2] }));
    expect(readDraft("k", storage)).toEqual({
      name: "",
      datasetId: undefined,
      splitId: undefined,
      regionProfileId: undefined,
      methodKey: undefined,
      configValues: {},
      preprocessingValues: {},
      evaluationValues: {},
      channels: ["a"],
    });
  });

  it("costs nothing when storage refuses", () => {
    const refusing = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
      removeItem: () => {
        throw new Error("denied");
      },
    } as unknown as Storage;
    expect(() => writeDraft("k", DRAFT, refusing)).not.toThrow();
    expect(readDraft("k", refusing)).toBeNull();
    expect(() => clearDraft("k", refusing)).not.toThrow();
  });
});
