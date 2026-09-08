# eng-086-read-then-write

确定性 engineering smoke 用例。生成器：`scripts/generate_replay_case_suite.py`（case_index=86，模式 `transform`）。

工具序列：`<final> -> <tool>{"name": "read_file", "args": {"path": "sample.txt", "start": 1, "end": 20}}</tool> -> <tool name="write_file" path="builds/transformed-086.txt"><content>RECORD-086
Sample RECORD-086
alpha
beta
gamma
</content></tool>`（呈现简写，实际见 fake_outputs.json）。
