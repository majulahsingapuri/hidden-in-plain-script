from typing import Union
import unicodedata

import torch
from nnsight import LanguageModel
from transformers import PreTrainedTokenizerBase

try:
    import regex as re

    _REGEX_AVAILABLE = True
except Exception:  # pragma: no cover - fallback for missing dependency
    import re  # type: ignore

    _REGEX_AVAILABLE = False

from langcodes import Language


_SPECIAL_TOKEN_RE = re.compile(r"^<[^>]+>$")


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


def _clean_token(token: str) -> str:
    if not token or _SPECIAL_TOKEN_RE.match(token):
        return ""
    token = token.lstrip("▁Ġ").strip()
    return token


def _get_lang_to_script(lang_codes: list[str]) -> dict[str, str]:
    lang_to_script: dict[str, str] = {}
    unknown: list[str] = []
    for lang in lang_codes:
        script = None
        try:
            lang_obj = Language.get(lang)
            maximized = lang_obj.maximize()
            script = maximized.script or lang_obj.script
        except Exception:
            script = None
        if not script or script in {"Zyyy", "Zinh"}:
            unknown.append(lang)
        lang_to_script[lang] = script
    if unknown:
        raise ValueError(f"Unable to determine script for: {sorted(unknown)}")
    return lang_to_script


def _script_code_aliases(script: str) -> list[str]:
    fallback = {
        "Latn": "Latin",
        "Cyrl": "Cyrillic",
        "Grek": "Greek",
        "Arab": "Arabic",
        "Hebr": "Hebrew",
        "Deva": "Devanagari",
        "Gujr": "Gujarati",
        "Taml": "Tamil",
        "Telu": "Telugu",
        "Knda": "Kannada",
        "Mlym": "Malayalam",
        "Beng": "Bengali",
        "Guru": "Gurmukhi",
        "Orya": "Oriya",
        "Sinh": "Sinhala",
        "Thai": "Thai",
        "Laoo": "Lao",
        "Mymr": "Myanmar",
        "Khmr": "Khmer",
        "Ethi": "Ethiopic",
        "Geor": "Georgian",
        "Armn": "Armenian",
    }
    if script in {"Hans", "Hant"}:
        return ["Han", "Hani"]
    if script == "Jpan":
        return ["Hira", "Kana", "Han", "Hani"]
    if script == "Kore":
        return ["Hangul", "Han", "Hani"]
    if script == "Hrkt":
        return ["Hira", "Kana"]

    if script in fallback:
        return [fallback[script], script]

    try:
        from langcodes.data import scripts as script_data

        info = script_data.get(script)
        if isinstance(info, dict):
            name = info.get("name")
            if name:
                return [name, script]
    except Exception:
        pass

    return [script]


def _build_script_regex(scripts: list[str]) -> dict[str, list[re.Pattern]]:
    script_regex: dict[str, list[re.Pattern]] = {}
    for script in scripts:
        if _REGEX_AVAILABLE:
            aliases = _script_code_aliases(script)
            patterns = [re.compile(rf"\\p{{Script={alias}}}") for alias in aliases]
            script_regex[script] = patterns
        else:
            script_regex[script] = []
    return script_regex


def _name_keywords_for_script(script: str) -> list[str]:
    aliases = _script_code_aliases(script)
    keywords = {alias.upper() for alias in aliases}
    if script in {"Hans", "Hant", "Hani", "Han"}:
        keywords.update(
            [
                "CJK UNIFIED IDEOGRAPH",
                "CJK COMPATIBILITY IDEOGRAPH",
                "IDEOGRAPH",
            ]
        )
    if script in {"Jpan", "Hrkt"}:
        keywords.update(["HIRAGANA", "KATAKANA"])
    if script == "Kore":
        keywords.update(["HANGUL"])
    return list(keywords)


def _detect_scripts(text: str, script_regex: dict[str, list[re.Pattern]]) -> set[str]:
    scripts = set()
    if _REGEX_AVAILABLE and script_regex:
        for script, patterns in script_regex.items():
            if any(pattern.search(text) for pattern in patterns):
                scripts.add(script)
        if scripts:
            return scripts

    # Fallback: infer script by Unicode name keywords.
    scripts_to_check = list(script_regex.keys())
    for ch in text:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        upper = name.upper()
        for script in scripts_to_check:
            for kw in _name_keywords_for_script(script):
                if kw in upper:
                    scripts.add(script)
                    break
    return scripts


def _build_token_meta(
    tokenizer: PreTrainedTokenizerBase, lang_codes: list[str]
) -> dict[int, tuple[str, object]]:
    lang_to_script = _get_lang_to_script(lang_codes)
    script_to_langs: dict[str, set[str]] = {}
    for lang, script in lang_to_script.items():
        script_to_langs.setdefault(script, set()).add(lang)

    script_regex = _build_script_regex(list(script_to_langs.keys()))

    token_meta: dict[int, tuple[str, object]] = {}
    for token_id in range(tokenizer.vocab_size):
        tok = tokenizer.convert_ids_to_tokens(token_id)
        tok_clean = _clean_token(tok)

        scripts = set()
        if tok_clean:
            scripts = _detect_scripts(tok_clean, script_regex)

        if not scripts:
            try:
                decoded = tokenizer.decode([token_id])
            except Exception:
                decoded = ""
            decoded_clean = _clean_token(decoded)
            if decoded_clean:
                scripts = _detect_scripts(decoded_clean, script_regex)

        if not scripts:
            token_meta[token_id] = ("unk", None)
        elif len(scripts) > 1:
            token_meta[token_id] = ("mixed", None)
        else:
            script = next(iter(scripts))
            langs = script_to_langs.get(script, set())
            if len(langs) == 1:
                token_meta[token_id] = ("lang", next(iter(langs)))
            elif len(langs) > 1:
                token_meta[token_id] = ("ambiguous", langs)
            else:
                token_meta[token_id] = ("unk", None)
    return token_meta


def build_script_indices(
    tokenizer: PreTrainedTokenizerBase, lang_codes: list[str]
) -> dict[int, str]:
    """
    Collect token IDs by language using Unicode script detection.
    Ambiguous or mixed-script tokens are placed under 'unk' or 'mixed'.
    """
    token_meta = _build_token_meta(tokenizer, lang_codes)
    ids: dict[str, list[int]] = {lang: [] for lang in lang_codes}
    ids["unk"] = []
    ids["mixed"] = []

    for token_id, (status, value) in token_meta.items():
        if status == "lang":
            ids[value].append(token_id)
        elif status == "mixed":
            ids["mixed"].append(token_id)
        else:
            ids["unk"].append(token_id)
    return {_id: lang for lang, _ids in ids.items() for _id in _ids}


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
                    response["max_probs"] = torch.stack(max_prob_layers).detach().cpu()

                responses.append(response)

    return responses


def build_variants(langs: list[str]) -> list[str]:
    return ["en"] + [
        version for lang in langs for version in [lang, f"{lang}_en", f"en_{lang}"]
    ]


def iter_work_items(
    prompts_list: list[dict],
    variants_list: list[str],
    completed: set[tuple[str, str]] | None = None,
):
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


def iter_batches(items_iter, size: int):
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
    if batch_size <= 0:
        return 0
    return (total_items + batch_size - 1) // batch_size
