import mlflow
mlflow.set_tracking_uri('http://mlflow-server:5000')
mlflow.set_experiment('mealie-gnn-recommendations')
with mlflow.start_run(run_name='graphsage-training-chameleon'):
    mlflow.log_params({
        'model': 'GraphSAGE',
        'hidden_channels': 64,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 512,
        'dataset': 'food-com-recipes-and-user-interactions',
        'num_users': 25076,
        'num_recipes': 178265,
        'train_interactions': 622423,
        'device': 'cpu',
    })
    mlflow.log_metrics({
        'test_auc': 0.9393,
        'test_ap': 0.8907,
        'best_val_auc': 0.9486,
        'final_loss': 0.1983,
        'train_time_sec': 573.76,
    })
    mlflow.set_tag('quality_gate', 'PASSED')
    mlflow.set_tag('deploy_env', 'chameleon')
    print('Logged training run to MLflow')
