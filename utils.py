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


def generate_trace(
    model: LanguageModel,
    prompt_texts: Union[str, list[str]],
    layers_path: str = "model.language_model.layers",
    norm_path: str = "model.language_model.norm",
) -> list[dict[str, Union[list[list[str]], torch.Tensor]]]:
    """Run a single prompt through Gemma and return the internal representations."""

    layers = resolve_attr_path(model, layers_path)
    norm = resolve_attr_path(model, norm_path)
    if isinstance(prompt_texts, str):
        prompt_texts = [prompt_texts]

    responses = []

    with model.trace() as tracer:
        for prompt_text in prompt_texts:
            with tracer.invoke(prompt_text) as invoker:
                probs_layers = []
                # store input tokens
                for layer_idx, layer in enumerate(layers):
                    # Process layer output through the model's head and layer normalization
                    layer_output = layer.output
                    layer_output_normed = model.lm_head(norm(layer_output))

                    # Apply softmax to obtain probabilities and save the result
                    layer_probs = torch.nn.functional.softmax(
                        layer_output_normed, dim=-1
                    ).save()
                    probs_layers.append(layer_probs)

                probs = torch.cat(probs_layers)

                # Find the maximum probability and corresponding tokens for each position
                max_probs, tokens = probs.max(dim=-1)

                # Decode token IDs to words for each layer
                words = [
                    [model.tokenizer.decode(t) for t in layer_tokens]
                    for layer_tokens in tokens
                ]

                # Access the 'input_ids' attribute of the invoker object to get the input words

                responses.append(
                    {
                        "prompt_text": prompt_text,
                        "words": words,
                        "max_probs": max_probs,
                        "tokens": tokens,
                    }
                )

    return responses
