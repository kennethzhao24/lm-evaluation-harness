import json
import torch
from collections import OrderedDict
from types import SimpleNamespace
from transformers import AutoConfig, PreTrainedTokenizerFast

from lm_eval.base import BaseLM

def check_for_weight_keys(ckpt):
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        if k[:6] == 'module':               
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


class OPT(BaseLM):
    def __init__(
        self,
        device="cuda",
        model_name=None,
        config_file=None,
        pretrained=None,
        tokenizer_file='/home/youpengzhao/code/accelarating-opt-main/data/20B_tokenizer.json',
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # load tokenizer
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        tokenizer.pad_token_id = 1
        self.tokenizer = tokenizer

        if model_name is not None:
            from .opt_models.opt_pytorch import OPTForCausalLM
            config = AutoConfig.from_pretrained(model_name)
        else:
            from .opt_models.dynamic_opt import OPTForCausalLM
            # load model config
            with open(config_file) as f:
                data = json.loads(f.read())
            config = SimpleNamespace(**data)
        
        # change vocab size to match 20B tokenizer
        config.vocab_size = 50277
        self.config = config

        self.opt = OPTForCausalLM(config)

        if pretrained is not None:
            ckpt = torch.load(pretrained, map_location='cpu')
            ckpt = check_for_weight_keys(ckpt)
            self.opt.load_state_dict(ckpt)
        
        self.opt.to(self.device)
        self.opt.eval()

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.opt(inps)[0][:, :, :50277]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.opt.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
