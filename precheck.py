"""Small diagnostic run for installation and mechanism sanity checking.

The output is not intended for scientific interpretation.
"""
from model import ModelConfig, AIPeerInteractionModel

cfg = ModelConfig(n_agents=40, episodes=12, seed=20260822)
model = AIPeerInteractionModel(cfg)
model.run()
summary = model.final_summary(tail=4)
print('Precheck completed successfully.')
print('Final summary:')
for key in ['final_peer_share','final_ai_share','final_hybrid_share','final_mean_quality','final_mean_knowledge']:
    if key in summary:
        print(f'  {key}: {summary[key]:.4f}')
