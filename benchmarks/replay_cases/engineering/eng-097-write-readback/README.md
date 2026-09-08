# eng-097-write-readback

确定性 engineering smoke 用例。生成器：`scripts/generate_replay_case_suite.py`（case_index=97，模式 `content-verified`）。

工具序列：`<final> -> <tool name="write_file" path="notes/scratch-097.txt"><content>Round-trip RECORD-097
alpha
beta
</content></tool> -> <tool>{"name": "read_file", "args": {"path": "notes/scratch-097.txt", "start": 1, "end": 20}}</tool>`（呈现简写，实际见 fake_outputs.json）。
