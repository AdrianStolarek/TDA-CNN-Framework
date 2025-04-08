import numpy as np
from sklearn.model_selection import KFold
from keras import callbacks
import mlflow


def cross_validation_objective(trial, X_data, y_data, input_shape, model_class, n_folds=5, experiment_name="default", optimize_target="accuracy"):
    """
    Returns:
        Average accuracy or negative loss across folds, depending on optimize_target
    """

    params = {
        'block_1_size': trial.suggest_int('block_1_size', 32, 128, step=8),
        'block_2_size': trial.suggest_int('block_2_size', 32, 256, step=16),
        'block_3_size': trial.suggest_int('block_3_size', 32, 256, step=32),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

    accuracies = []
    losses = []

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(params)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_data)):
            print(f"Training fold {fold+1}/{n_folds}")

            X_train, X_test = X_data[train_idx], X_data[test_idx]
            y_train, y_test = y_data[train_idx], y_data[test_idx]

            model = model_class(input_shape=input_shape, **params).model

            callbacks_es = [callbacks.EarlyStopping(
                monitor='accuracy', patience=8)]

            history = model.fit(
                X_train.astype('float32'),
                y_train.astype('int32'),
                epochs=40,
                batch_size=50,
                callbacks=callbacks_es,
                verbose=0
            )

            eval_results = model.evaluate(
                X_test.astype('float32'),
                y_test.astype('int32'),
                verbose=0
            )

            losses.append(eval_results[0])
            accuracies.append(eval_results[1])

            mlflow.log_metrics({
                f'fold_{fold+1}_loss': eval_results[0],
                f'fold_{fold+1}_accuracy': eval_results[1]
            })

        avg_loss = np.mean(losses)
        avg_accuracy = np.mean(accuracies)

        mlflow.log_metrics({
            'loss': avg_loss,
            'accuracy': avg_accuracy
        })

        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")

        if optimize_target == 'accuracy':
            return avg_accuracy
        else:
            return -avg_loss
