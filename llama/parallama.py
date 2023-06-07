from parallelformers.policies.base import Policy, Layer
from parallelformers.utils import AllReduceLinear
from llama.modeling_llama import LLaMADecoderLayer

class LLaMAPolicy(Policy): 
    @staticmethod
    def replace_arguments(config, world_size): 
        return {
            # hidden size
            "self_attn.hidden_size": config.hidden_size // world_size,
            # num attn heads
            "self_attn.num_heads": config.num_attention_heads // world_size,
        }
    
    @staticmethod
    def attn_qkv(): 
        return [
            Layer(
                weight="self_attn.q_proj.weight", 
            ), 
            Layer(
                weight="self_attn.k_proj.weight",
            ),
            Layer(
                weight="self_attn.v_proj.weight",
            ),
        ]
    
    @staticmethod
    def attn_out(): 
        return [
            Layer(
                weight="self_attn.o_proj.weight",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def mlp_in(): 
        return [
            Layer(
                weight="mlp.gate_proj.weight",
            ), 
            Layer(
                weight="mlp.up_proj.weight", 
            )
        ]
    
    @staticmethod
    def mlp_out():
        return [
            Layer(
                weight="mlp.down_proj.weight",
                replace=AllReduceLinear,
            )
        ]

    @staticmethod
    def original_layer_class(): 
        return LLaMADecoderLayer