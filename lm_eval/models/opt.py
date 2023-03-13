import json
import torch
from collections import OrderedDict
from types import SimpleNamespace
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast

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


def load_pretrained(ckpt, model):
    # save keys from model and pretrained weights
    model_keys = []
    for k, _ in model.state_dict().items():
        model_keys.append(k)
    weight_keys = []
    for k, _ in ckpt.items():
        weight_keys.append(k)
    # make sure dictionary size match
    assert len(weight_keys) == len(model_keys)
    
    new_weight = OrderedDict()
    for i in range(len(weight_keys)):
        # make sure weight shape size match
        assert ckpt[weight_keys[i]].shape == model.state_dict()[model_keys[i]].shape
        name = model_keys[i]
        new_weight[name] = model.state_dict()[model_keys[i]]
    model.load_state_dict(new_weight)


class OPT(BaseLM):
    def __init__(
        self,
        device="cuda",
        quantization=False,
        model_name=None,
        config_file=None,
        pretrained=None,
        tokenizer_file='./lm_eval/20B_tokenizer.json',
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        # assert isinstance(pretrained, str)
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
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.opt = OPTForCausalLM.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=config)
            print('Load {} model'.format(model_name))

        else:
            from .opt_models.dynamic_opt import OPTForCausalLM
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            tokenizer.pad_token_id = 1
            # load model config
            with open(config_file) as f:
                data = json.loads(f.read())
            config = SimpleNamespace(**data)        
            # change vocab size to match 20B tokenizer
            config.vocab_size = 50277
            self.opt = OPTForCausalLM(config)
            if pretrained is not None:
                ckpt = torch.load(pretrained, map_location='cpu')
                load_pretrained(ckpt, self.opt)
                # ckpt = check_for_weight_keys(ckpt)
                # self.opt.load_state_dict(ckpt)
            print('Load OPT Transformer model')


        self.config = config
        self.tokenizer = tokenizer

        if quantization:
            print('Perform 8-bit Quantization')
            # perform 8-bit quantization
            self.opt = torch.quantization.quantize_dynamic(
                self.opt, 
                {torch.nn.Linear}, 
                dtype=torch.qint8)
        
        self.opt.to(self._device)
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
