
#!/bin/bash
export API_KEY=
CONFIG_FILE="/raid/vinh/reward_model/configs/graph-qwen25_32b.yaml"

echo "Building graph with $CONFIG_FILE"
python /raid/vinh/reward_model/build_graph.py $CONFIG_FILE