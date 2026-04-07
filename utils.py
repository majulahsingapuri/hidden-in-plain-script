import unicodedata
from functools import lru_cache
from typing import Union

import torch
from nnsight import LanguageModel
from transformers import PreTrainedTokenizerBase


class ResourceMonitor:
    def __init__(self, gpu_index: int = 0):
        self._psutil = None
        self._pynvml = None
        self._gpu_handle = None
        self._nvml_init = False

        try:
            import psutil  # type: ignore

            self._psutil = psutil
            # Warm up CPU percent so subsequent calls are non-blocking.
            self._psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml_init = True
            if pynvml.nvmlDeviceGetCount() > gpu_index:
                self._pynvml = pynvml
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            if self._nvml_init:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            self._pynvml = None
            self._gpu_handle = None
            self._nvml_init = False

    def sample(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self._psutil is not None:
            metrics["cpu"] = float(self._psutil.cpu_percent(interval=None))
            metrics["ram"] = float(self._psutil.virtual_memory().percent)
        if self._pynvml is not None and self._gpu_handle is not None:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            metrics["gpu"] = float(util.gpu)
            metrics["vram"] = float((mem.used / mem.total) * 100) if mem.total else 0.0
        return metrics

    def tqdm_postfix(self) -> dict[str, str] | None:
        metrics = self.sample()
        if not metrics:
            return None
        postfix: dict[str, str] = {}
        if "cpu" in metrics:
            postfix["cpu%"] = f"{metrics['cpu']:.0f}"
        if "ram" in metrics:
            postfix["ram%"] = f"{metrics['ram']:.0f}"
        if "gpu" in metrics:
            postfix["gpu%"] = f"{metrics['gpu']:.0f}"
        if "vram" in metrics:
            postfix["vram%"] = f"{metrics['vram']:.0f}"
        return postfix

    def close(self):
        if self._pynvml is not None and self._nvml_init:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_init = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def resolve_attr_path(root: object, path: str) -> object:
    """Resolve dotted attribute paths with optional list indexing."""
    cur = root
    if not path:
        return cur
    for part in path.split("."):
        if not part:
            continue
        if part.endswith("]") and "[" in part:
            name, idx = part[:-1].split("[", 1)
            if name:
                cur = getattr(cur, name)
            cur = cur[int(idx)]
        elif part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _unwrap_layer_output(layer_output: object) -> object:
    """Return the hidden states from a layer output that may be a tuple/ModelOutput."""
    if isinstance(layer_output, (tuple, list)):
        return layer_output[0]
    if hasattr(layer_output, "last_hidden_state"):
        return getattr(layer_output, "last_hidden_state")
    if hasattr(layer_output, "hidden_states"):
        return getattr(layer_output, "hidden_states")
    return layer_output


def generate_trace(
    model: LanguageModel,
    prompt_texts: Union[str, list[str]],
    layers_path: str = "model.language_model.layers",
    norm_path: str = "model.language_model.norm",
    return_max_probs: bool = False,
) -> list[dict[str, Union[list[list[str]], torch.Tensor]]]:
    """Run a single prompt through Gemma and return the internal representations."""

    layers = resolve_attr_path(model, layers_path)
    norm = resolve_attr_path(model, norm_path)
    if isinstance(prompt_texts, str):
        prompt_texts = [prompt_texts]

    responses = []

    with model.trace() as tracer:
        for prompt_text in prompt_texts:
            with tracer.invoke(prompt_text):
                token_layers = []
                max_prob_layers = [] if return_max_probs else None

                # Store token IDs per layer without materializing full softmaxes.
                for layer_idx, layer in enumerate(layers):
                    layer_output = _unwrap_layer_output(layer.output)
                    logits = model.lm_head(norm(layer_output))

                    max_logits, token_ids = logits.max(dim=-1)
                    token_layers.append(token_ids)

                    if return_max_probs:
                        # Compute max softmax probability without storing the full distribution.
                        logsumexp = torch.logsumexp(logits, dim=-1)
                        max_prob_layers.append((max_logits - logsumexp).exp())

                tokens = torch.stack(token_layers)
                tokens_cpu = tokens.detach().cpu()
                if tokens_cpu.dim() == 3 and tokens_cpu.shape[1] == 1:
                    tokens_cpu = tokens_cpu[:, 0, :]

                words = [
                    [model.tokenizer.decode(int(t)) for t in layer_tokens]
                    for layer_tokens in tokens_cpu
                ]

                response = {
                    "prompt_text": prompt_text,
                    "words": words,
                    "tokens": tokens_cpu,
                }
                if return_max_probs and max_prob_layers is not None:
                    max_probs = torch.stack(max_prob_layers).detach().cpu()
                    if max_probs.dim() == 3 and max_probs.shape[1] == 1:
                        max_probs = max_probs[:, 0, :]
                    response["max_probs"] = max_probs

                responses.append(response)

    return responses


def build_variants(langs: list[str]) -> list[str]:
    """Return language pair variants used in experiments (en, lang, lang_en, en_lang)."""
    return ["en"] + [
        version for lang in langs for version in [lang, f"{lang}_en", f"en_{lang}"]
    ]


def iter_work_items(
    prompts_list: list[dict],
    variants_list: list[str],
    completed: set[tuple[str, str]] | None = None,
):
    """Yield pending work items for (prompt, variant) pairs, skipping completed ones."""
    completed = completed or set()
    for prompt in prompts_list:
        for variant in variants_list:
            key = (str(prompt.get("prompt_id")), str(variant))
            if key in completed:
                continue
            yield {
                "variant": variant,
                "prompt_text": prompt[variant],
                "prompt_en": prompt["en"],
                "prompt_id": prompt["prompt_id"],
            }


_EMOJI_RANGES: list[tuple[int, int]] = [
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x1F1E6, 0x1F1FF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
]

_SCRIPT_RANGES: list[tuple[str, list[tuple[int, int]]]] = [
    (
        "Latin",
        [
            (0x0041, 0x007A),
            (0x00C0, 0x00FF),
            (0x0100, 0x017F),
            (0x0180, 0x024F),
            (0x1E00, 0x1EFF),
            (0x2C60, 0x2C7F),
            (0xA720, 0xA7FF),
            (0xAB30, 0xAB6F),
        ],
    ),
    ("Greek", [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "Cyrillic",
        [
            (0x0400, 0x04FF),
            (0x0500, 0x052F),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA69F),
            (0x1C80, 0x1C8F),
        ],
    ),
    ("Armenian", [(0x0530, 0x058F)]),
    ("Hebrew", [(0x0590, 0x05FF)]),
    (
        "Arabic",
        [
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("Devanagari", [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("Bengali", [(0x0980, 0x09FF)]),
    ("Gurmukhi", [(0x0A00, 0x0A7F)]),
    ("Gujarati", [(0x0A80, 0x0AFF)]),
    ("Oriya", [(0x0B00, 0x0B7F)]),
    ("Tamil", [(0x0B80, 0x0BFF)]),
    ("Telugu", [(0x0C00, 0x0C7F)]),
    ("Kannada", [(0x0C80, 0x0CFF)]),
    ("Malayalam", [(0x0D00, 0x0D7F)]),
    ("Sinhala", [(0x0D80, 0x0DFF)]),
    ("Thai", [(0x0E00, 0x0E7F)]),
    ("Lao", [(0x0E80, 0x0EFF)]),
    ("Tibetan", [(0x0F00, 0x0FFF)]),
    ("Myanmar", [(0x1000, 0x109F), (0xAA60, 0xAA7F)]),
    ("Georgian", [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)]),
    ("Ethiopic", [(0x1200, 0x137F), (0x1380, 0x139F), (0x2D80, 0x2DDF)]),
    ("Khmer", [(0x1780, 0x17FF), (0x19E0, 0x19FF)]),
    ("Mongolian", [(0x1800, 0x18AF)]),
    (
        "Han",
        [
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
        ],
    ),
    ("Hiragana", [(0x3040, 0x309F)]),
    ("Katakana", [(0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D)]),
    ("Bopomofo", [(0x3100, 0x312F), (0x31A0, 0x31BF)]),
    (
        "Hangul",
        [
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
        ],
    ),
]


def _in_ranges(codepoint: int, ranges: list[tuple[int, int]]) -> bool:
    """Return True if codepoint falls within any (start, end) range."""
    for start, end in ranges:
        if start <= codepoint <= end:
            return True
    return False


@lru_cache(maxsize=4096)
def _char_script(ch: str) -> str:
    """Map a single Unicode character to a script label or bucket.

    Buckets:
    - Emoji: emoji codepoints.
    - Common: Unicode categories for punctuation, symbols, whitespace/separators,
      control chars, and digits.
    - Unknown: not covered by explicit script ranges.
    """
    codepoint = ord(ch)
    if _in_ranges(codepoint, _EMOJI_RANGES):
        return "Emoji"
    for script, ranges in _SCRIPT_RANGES:
        if _in_ranges(codepoint, ranges):
            return script
    category = unicodedata.category(ch)
    if category.startswith(("P", "S", "Z", "C", "N")):
        return "Common"
    return "Unknown"


def _normalize_token_text(token_text: str) -> str:
    """Normalize tokenizer markers (▁/Ġ/Ċ) into spaces/newlines for analysis."""
    if not token_text:
        return token_text
    # Common whitespace markers used by tokenizers.
    return token_text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")


def _token_text_to_script(
    token_text: str,
    *,
    emoji_label: str,
    common_label: str,
    unknown_label: str,
    mixed_label: str,
) -> str:
    """Map decoded token text to a script label.

    Rules:
    - If all non-Common chars belong to one script, return that script.
    - If multiple scripts appear, return the mixed label.
    - If only Common chars appear, return the common label.
    - If no chars appear, return the unknown label.
    """
    token_text = _normalize_token_text(token_text)

    counts: dict[str, int] = {}
    common_count = 0
    for ch in token_text:
        script = _char_script(ch)
        if script == "Common":
            common_count += 1
            continue
        counts[script] = counts.get(script, 0) + 1

    if not counts:
        if common_count > 0:
            return common_label
        return unknown_label

    if len(counts) == 1:
        only_script = next(iter(counts))
        return emoji_label if only_script == "Emoji" else only_script

    return mixed_label


def token_id_to_script(
    token_id: int,
    tokenizer: PreTrainedTokenizerBase,
    *,
    special_label: str = "Special",
    emoji_label: str = "Emoji",
    common_label: str = "Common",
    unknown_label: str = "Unknown",
    mixed_label: str = "Mixed",
) -> str:
    """Map a token id to a Unicode script label based on decoded text.

    Labels:
    - Special: tokenizer special tokens (e.g., <end_of_turn>).
    - Common: punctuation, symbols, whitespace/separators, control chars, and digits.
    - Emoji: emoji codepoints.
    - Mixed: token contains multiple scripts.
    - Unknown: no characters or unsupported codepoints.
    """
    if token_id in tokenizer.all_special_ids:
        return special_label

    decoded = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    token_text = (
        decoded
        if decoded
        else tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False)
    )
    return _token_text_to_script(
        token_text,
        emoji_label=emoji_label,
        common_label=common_label,
        unknown_label=unknown_label,
        mixed_label=mixed_label,
    )


def build_token_script_map(
    tokenizer: PreTrainedTokenizerBase,
    *,
    special_label: str = "Special",
    emoji_label: str = "Emoji",
    common_label: str = "Common",
    unknown_label: str = "Unknown",
    mixed_label: str = "Mixed",
) -> dict[int, str]:
    """Map every token id in a tokenizer to a Unicode script label.

    Requires a vocabulary mapping via tokenizer.get_vocab() or tokenizer.vocab.

    Labels:
    - Special: tokenizer special tokens (e.g., <end_of_turn>).
    - Common: punctuation, symbols, whitespace/separators, control chars, and digits.
    - Emoji: emoji codepoints.
    - Mixed: token contains multiple scripts.
    - Unknown: no characters or unsupported codepoints.
    """
    script_map: dict[int, str] = {}
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)

    vocab: dict[str, int] | None = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocab = get_vocab()
    elif isinstance(getattr(tokenizer, "vocab", None), dict):
        vocab = tokenizer.vocab  # type: ignore[assignment]

    if not vocab:
        raise ValueError("Tokenizer does not expose a vocabulary mapping.")

    for token_text, token_id in vocab.items():
        if token_id in special_ids:
            script_map[token_id] = special_label
            continue
        script_map[token_id] = _token_text_to_script(
            token_text,
            emoji_label=emoji_label,
            common_label=common_label,
            unknown_label=unknown_label,
            mixed_label=mixed_label,
        )

    return script_map


def iter_batches(items_iter, size: int):
    """Yield consecutive batches of a given size from an iterator."""
    batch = []
    for item in items_iter:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_work_items(
    prompts_list: list[dict],
    variants_list: list[str],
    completed: set[tuple[str, str]] | None = None,
) -> int:
    """Count pending work items for (prompt, variant) pairs."""
    completed = completed or set()
    count = 0
    for prompt in prompts_list:
        for variant in variants_list:
            key = (str(prompt.get("prompt_id")), str(variant))
            if key in completed:
                continue
            count += 1
    return count


def count_batches(total_items: int, batch_size: int) -> int:
    """Return the number of batches needed for total_items at batch_size."""
    if batch_size <= 0:
        return 0
    return (total_items + batch_size - 1) // batch_size
