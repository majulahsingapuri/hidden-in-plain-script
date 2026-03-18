import torch
from nnsight import LanguageModel


def generate_response(
    model: LanguageModel,
    prompt_text: str,
    max_new_tokens: int = 300,
    temperature: float = 0.9,
    top_p: int = 0,
):
    """Run a single prompt through Gemma and return the generated text."""
    with model.generate(
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p_threshold=top_p,
    ):
        resp = model.tokenizer.decode(model.generator.output[0])

    return resp


def generate_trace(model: LanguageModel, prompt_text: str):
    """Run a single prompt through Gemma and return the internal representations."""
    probs_layers = []

    layers = model.model.language_model.layers

    with model.trace() as tracer:
        with tracer.invoke(prompt_text) as invoker:
            # store input tokens
            input_tokens = invoker.inputs.save()
            for layer_idx, layer in enumerate(layers):
                # Process layer output through the model's head and layer normalization
                layer_output = layer.output
                layer_output_normed = model.lm_head(
                    model.model.language_model.norm(layer_output)
                )

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
            input_words = [
                model.tokenizer.decode(t) for t in input_tokens[1]["input_ids"][0]
            ]

            response = {
                "input_words": input_words,
                "words": words,
                "max_probs": max_probs,
                "tokens": tokens,
            }

    return response
