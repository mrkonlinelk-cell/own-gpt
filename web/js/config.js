// Repo this site belongs to. Used by the Train page to call the GitHub API.
export const REPO_OWNER = "mrkonlinelk-cell";
export const REPO_NAME = "own-gpt";

export const MODEL_URL = new URL("../model/model.onnx", import.meta.url).href;
export const TOKENIZER_URL = new URL("../model/tokenizer.json", import.meta.url).href;
export const META_URL = new URL("../model/meta.json", import.meta.url).href;

export const DEFAULT_TEMPERATURE = 0.8;
export const DEFAULT_TOP_K = 40;
export const DEFAULT_MAX_TOKENS = 80;
export const DEFAULT_BLOCK_SIZE = 256;
export const END_OF_TEXT = "<|endoftext|>";
