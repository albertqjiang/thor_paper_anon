import time
import logging
import numpy as np


def generate_target(
    model,
    input_with_sep,
    batch_size,
    gen_length,
    min_input_seq_len=64,
    sampler_options=None,
    return_logits=False,
):
    logging.debug("inference:generate_target - Begin function")
    if sampler_options is None:
        sampler_options = {
            "top_p": np.ones(batch_size) * 0.9,
            "temp": np.ones(batch_size) * 0.75,
        }

    start = time.time()

    pad_len = min_input_seq_len
    while pad_len < len(input_with_sep):  # speed up jit compilation
        pad_len *= 2
    padded_proof_state = np.pad(
        input_with_sep, ((pad_len - len(input_with_sep), 0),)
    ).astype(np.uint32)
    batched_tokens = np.repeat(
        np.array(padded_proof_state)[None, :], batch_size, axis=0
    )

    input_len = np.ones(batch_size, dtype=np.uint32) * len(input_with_sep)

    logging.info(f"inference:generate_target - Sequence ready, seq_len: {pad_len}")
    output = model.generate(
        batched_tokens,
        input_len,
        gen_length=gen_length,
        sampler_options=sampler_options,
        return_logits=return_logits,
    )

    res = output[1][0][:, :, 0]

    print(f"Seq len: {pad_len}, completion done in {time.time() - start:06}s")
    logging.info(f"inference:generate_target - Completion done in {time.time() - start:06}s")

    if return_logits:
        logits = output[1][-1]
        return res, logits
    return res
