import torch
from transformers import PegasusForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class PetPegasusForConditionalGeneration(PegasusForConditionalGeneration):

    @staticmethod
    def _get_len_to_output_map(all_model_kwargs):
        all_len_to_output_map = []
        for model_kwargs in all_model_kwargs:
            if 'output_prefix_ids' not in model_kwargs:
                all_len_to_output_map.append({})
            else:
                output_prefix_ids = model_kwargs['output_prefix_ids']
                all_len_to_output_map.append({idx + 1: token_id for idx, token_id in enumerate(output_prefix_ids)})
        return all_len_to_output_map

    def _generate_no_beam_search(
            self,
            input_ids,
            cur_len,
            max_length,
            min_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            no_repeat_ngram_size,
            bad_words_ids,
            pad_token_id,
            eos_token_id,
            batch_size,
            attention_mask,
            use_cache,
            model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        If there are m patterns and n examples, this implementation expects the inputs (input_ids, attention_mask and model_kwargs) to be
        grouped as follows: [input_1_pattern_1, input_1_pattern_2, ..., input_1_pattern_m, input_2_pattern_1, ..., input_n_pattern_m]
        """

        num_patterns = len(model_kwargs['output_prefix_ids']) if model_kwargs['output_prefix_ids'] else 1
        joint_decoding = model_kwargs.get('joint_decoding', False)

        kwargs_keys = {'encoder_outputs', 'output_prefix_ids'}
        assert kwargs_keys.issubset(set(model_kwargs.keys())), f"Got kwargs {set(model_kwargs.keys())} but expected {kwargs_keys}"

        assert input_ids.shape[0] % num_patterns == attention_mask.shape[0] % num_patterns == \
               model_kwargs['encoder_outputs'].last_hidden_state.shape[0] % num_patterns == 0

        grouped_attention_masks = [attention_mask[start::num_patterns] for start in range(num_patterns)]
        grouped_input_ids = [input_ids[start::num_patterns] for start in range(num_patterns)]

        grouped_model_kwargs = [{
            'encoder_outputs': BaseModelOutput(last_hidden_state=model_kwargs['encoder_outputs'].last_hidden_state[start::num_patterns]),
            'output_prefix_ids': model_kwargs['output_prefix_ids'][start]
        } for start in range(num_patterns)]

        grouped_len_to_output = PetPegasusForConditionalGeneration._get_len_to_output_map(grouped_model_kwargs)
        prefix_lengths = [len(len_to_output) for len_to_output in grouped_len_to_output]

        assert len(grouped_input_ids) == len(grouped_attention_masks) == len(grouped_model_kwargs) == len(prefix_lengths)

        # length of generated sentences / unfinished sentences
        grouped_unfinished_sents = [input_ids.new(input_ids.shape[0]).fill_(1) for input_ids in grouped_input_ids]

        grouped_past = [None for _ in grouped_input_ids]
        decoding_started = False
        indices_to_process = []

        cur_prefix_len = 1
        while cur_len < max_length:

            if not decoding_started:
                indices_to_process = [idx for idx, prefix_length in enumerate(prefix_lengths) if prefix_length > 0]
                if not indices_to_process:
                    decoding_started = True
                    indices_to_process = [idx for idx, _ in enumerate(prefix_lengths)]

            prefix_lengths = [max(0, prefix_length - 1) for prefix_length in prefix_lengths]

            # if not decoding_started, there are some prefixes left to be processed
            # otherwise, we are now aligned so we can start with regular decoding
            grouped_next_token_logits = {}
            for idx in indices_to_process:

                model_inputs = self.prepare_inputs_for_generation(
                    grouped_input_ids[idx], past=grouped_past[idx], attention_mask=grouped_attention_masks[idx], use_cache=use_cache,
                    **grouped_model_kwargs[idx]
                )

                outputs = self(**model_inputs, return_dict=True)
                grouped_next_token_logits[idx] = outputs.logits[:, -1, :]

                if "past_key_values" in outputs:
                    grouped_past[idx] = outputs.past_key_values
                elif "mems" in outputs:
                    grouped_past[idx] = outputs.mems

            finished_indices = set()

            next_token = None

            if decoding_started and joint_decoding:
                next_token_logits = torch.stack(tuple(grouped_next_token_logits.values()), dim=0).sum(dim=0)
                next_token = torch.argmax(next_token_logits, dim=-1)

            for idx in indices_to_process:
                if not decoding_started:
                    assert cur_prefix_len in grouped_len_to_output[idx].keys()
                    next_token = torch.tensor([grouped_len_to_output[idx][cur_prefix_len]] * grouped_input_ids[idx].shape[0],
                                              device=grouped_input_ids[idx].device)
                elif not joint_decoding:
                    next_token = torch.argmax(grouped_next_token_logits[idx], dim=-1)

                # update generations and finished sentences
                if eos_token_id is not None:
                    # pad finished sentences if eos_token_id exists
                    tokens_to_add = next_token * grouped_unfinished_sents[idx] + (pad_token_id) * (1 - grouped_unfinished_sents[idx])
                else:
                    tokens_to_add = next_token

                grouped_input_ids[idx] = torch.cat([grouped_input_ids[idx], tokens_to_add.unsqueeze(-1)], dim=-1)

                if eos_token_id is not None:
                    # unfinished_sents is set to zero if eos in sentence
                    eos_in_sents = tokens_to_add == eos_token_id
                    grouped_unfinished_sents[idx].mul_((~eos_in_sents).long())

                # stop when there is a </s> in each sentence, or if we exceed the max length
                if grouped_unfinished_sents[idx].max() == 0:
                    finished_indices.add(idx)

            for fi in finished_indices:
                indices_to_process.remove(fi)

            if not indices_to_process:
                break

            if decoding_started:
                cur_len = cur_len + 1
            else:
                cur_prefix_len = cur_prefix_len + 1

        # regroup by sentence instead of by pattern id
        num_sentences = grouped_input_ids[0].shape[0]
        results = []
        for sent_idx in range(num_sentences):
            sent_results = []
            for pattern_idx in range(num_patterns):
                sent_results.append(grouped_input_ids[pattern_idx][sent_idx])
            results.append(sent_results)

        return results

    def _generate_beam_search(**kwargs):
        raise NotImplementedError("Beam search is not implemented for joint decoding")
