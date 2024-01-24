# Utilities to train an ensemble

def train_ensemble(train_fn, models, optimizers, *args):
    for i_model, (model, optimizer) in enumerate(zip(models, optimizers)):
        print("Training model", i_model, "out of", len(models))
        train_fn(model, optimizer, *args)
