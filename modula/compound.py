from modula.abstract import *
from modula.atom import *
from modula.bond import *

def MLP(output_dim, input_dim, width, depth):
    m = Linear(output_dim, width) @ ReLU()
    for _ in range(depth-2):
        m = m @ Linear(width, width) @ ReLU()
    return m @ Linear(width, input_dim)

def Attention(num_heads, d_embed, d_query, d_value, softmax_scale, causal):
    """Multi-head attention"""
    Q = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = SplitIntoHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = SplitIntoHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, num_heads * d_value) @ MergeHeads()

    AttentionScores = Softmax(softmax_scale) @ CausalMask() @ AttentionQK() @ Rope(d_query) @ (Q, K)
    return W @ (1/3 * ApplyAttentionScores()) @ (V, AttentionScores)

def GPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    att = Attention(num_heads, d_embed, d_query, d_value, attention_scale, causal=True)
    mlp = Linear(d_embed, 4*d_embed) @ GeLU() @ Linear(4*d_embed, d_embed)
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed)

    return out @ blocks @ embed

def OrthogonalAttention(num_heads, d_embed, softmax_scale, causal):
    """
    Orthogonal attention uses 3-tensors for Q, K, V to make the input and output dimensions explicitly equal.
    """
    Q = TransposeHeads() @ HeadedLinear(num_heads, d_embed, d_embed)
    K = TransposeHeads() @ HeadedLinear(num_heads, d_embed, d_embed)
    V = TransposeHeads() @ HeadedLinear(num_heads, d_embed, d_embed)
    W = HeadedAttentionOut(num_heads, d_embed, d_embed)

    AttentionScores = Softmax(softmax_scale) @ CausalMask() @ AttentionQK() @ Rope(d_embed) @ (Q, K)
    return (1/3) * W @ (V, AttentionScores)

def OrthogonalGPT(vocab_size, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5, attention_scale=1.0, final_scale=1.0):
    embed = Embed(d_embed, vocab_size)
    embed.tare()

    att = OrthogonalAttention(num_heads, d_embed, d_query=d_embed, d_value=d_embed, attention_scale=attention_scale, causal=True)
    mlp = Linear(d_embed, d_embed) @ GeLU() @ Linear(d_embed, d_embed)
    att_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ att_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    out = final_scale * Linear(vocab_size, d_embed)

    return out @ blocks @ embed