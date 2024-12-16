class DefaultModelAssessment:
    def __init__(self, render_dir=None):
        self.render_dir = render_dir

    def evaluate_model(self, model, env, ticker):
        total_positions = 0
        won_positions = 0
        cumulated_reward = 0
        total_reward = 1

        obs, info = env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward *= (1 + reward/100)

            if action != 0:
                total_positions += 1
                if reward > 0:
                    won_positions += 1
                    cumulated_reward += reward/100

            if done or truncated:
                break
        
        if self.render_dir:
            env.save_render(f"{self.render_dir}/{ticker}")

        return {
            "ticker": ticker,
            "total_reward": total_reward,
            "compound_return": total_reward - 1,
            "profitable": total_reward - 1 > 0,
            "total_positions": total_positions,
            "win_rate": won_positions / total_positions if total_positions > 0 else 0,
            "avg_position_return": cumulated_reward / won_positions if won_positions > 0 else 0,
        }
    

class DefaultGroupAssessment:
    def __init__(self, model_assessment, env_factory):
        self.model_assessment = model_assessment
        self.env_factory = env_factory

    def evaluate_group(self, name, env_specs, model):
        num_profitables = 0
        total_reward = 0

        results = []
        for specs in env_specs:
            ticker, frame_bounds = specs
            env = self.env_factory(ticker, frame_bounds, no_action_punishment=0)
            result = self.model_assessment.evaluate_model(model, env, ticker)
            
            if result['profitable']:
                num_profitables += 1

            total_reward += result['compound_return']

            results.append(result)

        return {
            "name": name,
            "details": results,
            "num_profitables": num_profitables,
            "perc_profitables": num_profitables / len(env_specs),
            "total_reward": total_reward,
            "avg_reward": total_reward / len(env_specs),
        }
    

def model_is_profitable(
        model,
        group_assessment,
        group_name,
        env_specs,
        max_non_profitable_allowed=1
    ):
    assessment = group_assessment.evaluate_group(group_name, env_specs, model)
    num_non_profitables = 0
    for detail in assessment['details']:
        if detail["win_rate"] >= 0.65 and detail["avg_position_return"] >= 0.02:
            continue
        num_non_profitables += 1
    return num_non_profitables <= max_non_profitable_allowed