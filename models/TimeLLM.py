from math import sqrt

import torch
import torch.nn as nn

# Import AutoConfig, AutoModel, AutoTokenizer for HyperCLOVAX
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.rsi_val = 0 
        self.upper_band = 0 
        self.lower_band = 0

        # Define default max length for HyperCLOVAX based on its stated context window
        self.tokenizer_max_length_hyperclovax = 16384 # 16K tokens for HyperCLOVAX

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        # --- Start HyperCLOVAX additions ---
        elif configs.llm_model == 'HyperCLOVAX':
            self.hyperclovax_model_name = 'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'
            
            self.hyperclovax_config = AutoConfig.from_pretrained(self.hyperclovax_model_name, trust_remote_code=True)
            self.hyperclovax_config.num_hidden_layers = configs.llm_layers
            self.hyperclovax_config.output_attentions = True
            self.hyperclovax_config.output_hidden_states = True
            
            # Update model_max_length in config, ensuring it's not the '1e30' default
            # It's explicitly stated as 16k in its model card, so we'll enforce that.
            self.hyperclovax_config.max_position_embeddings = self.tokenizer_max_length_hyperclovax
            
            try:
                self.llm_model = AutoModel.from_pretrained(
                    self.hyperclovax_model_name,
                    trust_remote_code=True,
                    local_files_only=True, # Try local first
                    config=self.hyperclovax_config,
                )
            except EnvironmentError:
                print(f"Local model files for {self.hyperclovax_model_name} not found. Attempting to download...")
                print("Note: HyperCLOVAX models are gated. Ensure you are logged into Hugging Face and have accepted terms.")
                self.llm_model = AutoModel.from_pretrained(
                    self.hyperclovax_model_name,
                    trust_remote_code=True,
                    local_files_only=False, # Allow download
                    config=self.hyperclovax_config,
                )

            try:
                # Use AutoTokenizer for flexibility with HyperCLOVAX
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hyperclovax_model_name,
                    trust_remote_code=True,
                    local_files_only=True, # Try local first
                    # Explicitly set model_max_length here to avoid the OverflowError
                    model_max_length=self.tokenizer_max_length_hyperclovax 
                )
            except EnvironmentError:
                print(f"Local tokenizer files for {self.hyperclovax_model_name} not found. Attempting to download them..")
                print("Note: HyperCLOVAX models are gated. Ensure you are logged into Hugging Face and have accepted terms.")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hyperclovax_model_name,
                    trust_remote_code=True,
                    local_files_only=False, # Allow download
                    # Explicitly set model_max_length here to avoid the OverflowError
                    model_max_length=self.tokenizer_max_length_hyperclovax
                )
        # --- End HyperCLOVAX additions ---
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # For models that might not have an EOS token or a default pad token (e.g., some BERT variants)
            # HyperCLOVAX also seems to use <|endoftext|> as its EOS, but this is a safe fallback.
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            # Ensure the embedding layer is resized if new tokens are added
            self.llm_model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            # Updated description to be more general for time series, as ETT is just one example.
            self.description = 'The input data represents multivariate time series, capturing dynamic trends and periodic patterns across various variables. This model aims to forecast future values based on historical observations.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000 # This 'num_tokens' needs to be carefully chosen based on what it's mapping to.
                               # If it's a fixed size for the mapping layer, it's fine.
                               # But if it represents a subset of vocabulary, it might need adjustment based on actual vocab.
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        moving_averages_5 = self.calculate_moving_average(x_enc, window=5)
        moving_averages_20 = self.calculate_moving_average(x_enc, window=20)
        rsi_vals = self.calculate_rsi(x_enc, window=14)
        middle_bands, upper_bands, lower_bands = self.calculate_bollinger_bands(x_enc, window=20, num_std_dev=2)

        
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            moving_averages_5_str = str(moving_averages_5[b].tolist()) if isinstance(moving_averages_5[b], torch.Tensor) else str(moving_averages_5[b])
            moving_averages_20_str = str(moving_averages_20[b].tolist()) if isinstance(moving_averages_20[b], torch.Tensor) else str(moving_averages_20[b])
            
            rsi_val_str = str(rsi_vals[b].tolist()) if isinstance(rsi_vals[b], torch.Tensor) else str(rsi_vals[b])
            upper_band_str = str(upper_bands[b].tolist()) if isinstance(upper_bands[b], torch.Tensor) else str(upper_bands[b])
            lower_band_str = str(lower_bands[b].tolist()) if isinstance(lower_bands[b], torch.Tensor) else str(lower_bands[b])

            rsi_state = (
                "과매수 상태" if rsi_vals[b] > 70 else
                "과매도 상태" if rsi_vals[b] < 30 else
                "중립 상태"
            )
            
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}, "
                f"5-period moving average: {moving_averages_5_str}, "
                f"20-period moving average: {moving_averages_20_str}, "
                f"RSI (14-period): {rsi_val_str} ({rsi_state}), "
                f"Bollinger Bands (20-period, 2 std dev): upper band {upper_band_str}, lower band {lower_band_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # Dynamically set max_length for the tokenizer based on the LLM model
        if hasattr(self, 'tokenizer_max_length_hyperclovax') and self.llm_model.name_or_path == self.hyperclovax_model_name:
            current_max_length = self.tokenizer_max_length_hyperclovax
        elif hasattr(self.llm_model.config, 'max_position_embeddings'):
            current_max_length = self.llm_model.config.max_position_embeddings
        elif hasattr(self.llm_model.config, 'n_positions'): # For GPT-2
            current_max_length = self.llm_model.config.n_positions
        else:
            # Fallback to a reasonable default or raise error if max_length cannot be determined
            current_max_length = 2048 # A common default
            print(f"Warning: Could not determine specific max_length for {self.llm_model.name_or_path}. Using default: {current_max_length}")


        # Pass the determined max_length to the tokenizer
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=current_max_length).input_ids
        
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    def calculate_moving_average(self, data, window):
        # Implement your moving average calculation here
        # Example: Simple Moving Average
        if data.shape[1] < window:
            return torch.zeros(data.shape[0]) # Or handle error/edge case
        return torch.mean(data[:, -window:, 0], dim=1) # Assuming last dimension is 1

    def calculate_rsi(self, data, window=14):
        # Implement RSI calculation (usually on closing prices or a single series)
        # This is a simplified placeholder.
        # A proper RSI calculation involves average gains and losses over the window.
        if data.shape[1] < window + 1: # Need at least window + 1 points for changes
            return torch.zeros(data.shape[0]) + 50 # Return neutral if not enough data

        # Calculate price changes
        deltas = data[:, 1:, 0] - data[:, :-1, 0]
        gains = torch.relu(deltas)
        losses = torch.relu(-deltas)

        avg_gain = torch.zeros(data.shape[0])
        avg_loss = torch.zeros(data.shape[0])

        for i in range(data.shape[0]):
            current_gains = gains[i, :]
            current_losses = losses[i, :]

            # Initial average gain/loss for the first 'window' periods
            if current_gains.shape[0] >= window:
                avg_gain[i] = torch.mean(current_gains[:window])
                avg_loss[i] = torch.mean(current_losses[:window])

                # Subsequent periods use Wilder's smoothing
                for j in range(window, current_gains.shape[0]):
                    avg_gain[i] = (avg_gain[i] * (window - 1) + current_gains[j]) / window
                    avg_loss[i] = (avg_loss[i] * (window - 1) + current_losses[j]) / window
            else:
                avg_gain[i] = torch.mean(current_gains) if current_gains.numel() > 0 else 0.
                avg_loss[i] = torch.mean(current_losses) if current_losses.numel() > 0 else 0.

        rs = torch.where(avg_loss == 0, torch.tensor(float('inf')), avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi


    def calculate_bollinger_bands(self, data, window=20, num_std_dev=2):
        # Implement Bollinger Bands calculation
        # Middle Band: Simple Moving Average
        # Upper Band: Middle Band + (num_std_dev * Standard Deviation)
        # Lower Band: Middle Band - (num_std_dev * Standard Deviation)
        if data.shape[1] < window:
            return torch.zeros(data.shape[0]), torch.zeros(data.shape[0]), torch.zeros(data.shape[0])

        middle_band = torch.mean(data[:, -window:, 0], dim=1)
        std_dev = torch.std(data[:, -window:, 0], dim=1)

        upper_band = middle_band + (num_std_dev * std_dev)
        lower_band = middle_band - (num_std_dev * std_dev)
        return middle_band, upper_band, lower_band


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
