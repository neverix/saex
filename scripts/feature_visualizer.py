import gradio as gr
import pyarrow.parquet as pq
import pyarrow.compute as pc
from transformers import AutoTokenizer
import os


token_table = pq.read_table("weights/tokens.parquet")
cache_path = "weights/caches"
parquets = os.listdir(cache_path)
TOKENIZER = "microsoft/Phi-3-mini-4k-instruct"
token_range = 32
with gr.Blocks() as demo:
    feature_table = gr.State(None)

    tokenizer_name = gr.Textbox(TOKENIZER)
    dropdown = gr.Dropdown(parquets)
    feature_input = gr.Number(0)

    frame = gr.Dataframe()

    def update(cache_name, feature, tokenizer_name):
        if cache_name is None:
            return
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        table = pq.read_table(f"{cache_path}/{cache_name}")
        table_feat = table.filter(pc.field("feature") == feature).to_pandas()
        table_feat["text"] = table_feat["token"].apply(
            lambda x: tokenizer.decode(token_table[max(0, x - token_range):x+1]["tokens"].to_numpy()))
        return table_feat

    dropdown.change(update, [dropdown, feature_input, tokenizer_name], [frame])
    feature_input.change(update, [dropdown, feature_input, tokenizer_name], [frame])


if __name__ == "__main__":
    demo.launch(share=True)
