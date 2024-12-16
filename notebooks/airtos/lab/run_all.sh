#! bin/bash

echo "" > results.txt


python rate_agent.py models/a2c_1 >> results.txt
python rate_agent.py models/a2c_2 >> results.txt
python rate_agent.py models/a2c_3 >> results.txt
python rate_agent.py models/a2c_4 >> results.txt
python rate_agent.py models/a2c_5 >> results.txt
python rate_agent.py models/dqn_2 >> results.txt
python rate_agent.py models/dqn_3 >> results.txt
python rate_agent.py models/dqn_4 >> results.txt
python rate_agent.py models/dqn_5 >> results.txt
python rate_agent.py models/ppo_2 >> results.txt
python rate_agent.py models/ppo_3 >> results.txt
python rate_agent.py models/ppo_4 >> results.txt
python rate_agent.py models/ppo_5 >> results.txt
python rate_agent.py models/qr_dqn_2 >> results.txt
python rate_agent.py models/qr_dqn_3 >> results.txt
python rate_agent.py models/qr_dqn_4 >> results.txt
