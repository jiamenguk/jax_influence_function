import copy
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

from jax import jvp
from jax.tree_util import tree_flatten, tree_leaves

from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax.scipy.special import logsumexp
from jax.experimental import optimizers

from torch.utils import data
from functools import partial


dataset = json.loads(uploaded['chatbot_intent_classifier_data.json'])

proc_dataset = []
for i, unit in enumerate(dataset):
    proc_unit = copy.deepcopy(unit)
    proc_unit['data'] = json.loads(proc_unit['data'])
    proc_dataset.append(proc_unit)

completed_set = [i for i in proc_dataset if i['complete']]
training_set = [i for i in completed_set if i['usage'] == 'training']
testing_set = [i for i in completed_set if i['usage'] == 'testing']
validation_set = [i for i in completed_set if i['usage'] == 'validation']
classes = list(set([i['data']['intent'] for i in completed_set]))
n_targets = len(classes)

class_map = {v: i for i, v in enumerate(classes)}
inv_class_map = {v: k for k, v in class_map.items()}

# Get the full train dataset (for checking accuracy while training)
train_sentences = [i['data']['text'] for i in training_set]
train_sentences = sentence_model.encode(train_sentences)
train_labels = [i['data']['intent'] for i in training_set]
train_labels = np.array([class_map[i] for i in train_labels])
train_labels = one_hot(train_labels, n_targets)

# Get full test dataset
validation_sentences = [i['data']['text'] for i in validation_set]
validation_sentences = sentence_model.encode(validation_sentences)
validation_labels = [i['data']['intent'] for i in validation_set]
validation_labels = np.array([class_map[i] for i in validation_labels])
validation_labels = one_hot(validation_labels, n_targets)

layer_sizes = [768, 128, 12]
step_size = 0.001
num_epochs = 20
batch_size = 64

training_dataset = SentenceDataset(training_set)
training_generator = SentenceLoader(training_dataset, shuffle=True, batch_size=batch_size, num_workers=0)

params = init_network_params(layer_sizes, random.PRNGKey(42))
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in training_generator:
        y = one_hot(y, n_targets)
        params, opt_state = update(epoch, get_params(opt_state), x, y, opt_state)
    epoch_time = time.time() - start_time

    train_acc = accuracy(get_params(opt_state), train_sentences, train_labels)
    test_acc = accuracy(get_params(opt_state), validation_sentences, validation_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))

testing_sentences = [i['data']['text'] for i in testing_set]
testing_sentences = sentence_model.encode(testing_sentences)
testing_labels = [i['data']['intent'] for i in testing_set]
testing_labels = np.array([class_map[i] for i in testing_labels])

predicted_class = jnp.argmax(batched_predict(params, testing_sentences), axis=1)
logprob = batched_predict(params, testing_sentences)
pred_prob = jnp.exp(logprob)

wrong_indices = [i for i, x in enumerate(testing_labels != predicted_class) if x]
print(wrong_indices)


idx = 6

testing_id = wrong_indices[idx]
print("Testing id:", testing_id)

label_class = class_map[testing_set[testing_id]['data']['intent']]
print("Sentence:", testing_set[testing_id]['data']['text'])
print("True label:", testing_set[testing_id]['data']['intent'])
print("True label prob:", pred_prob[testing_id][label_class])
print("Predicted label:", inv_class_map[int(predicted_class[testing_id])])
print("Predicted prob:", jnp.max(pred_prob[testing_id]))

z_loader = SentenceLoader(training_dataset, shuffle=True, batch_size=1, num_workers=0)

s_test = get_s_test(testing_sentences[testing_id], testing_labels[testing_id], params, z_loader)

z_loader = SentenceLoader(training_dataset, shuffle=False, batch_size=8, num_workers=0)

influences = []
for i, (x, t) in enumerate(z_loader):
    t = one_hot(t, n_targets)
    z = [i for i in zip(x, t)]
    tmp_influence = vmap(partial(get_influence, params=params, s_test=s_test, N=len(training_set)), in_axes=(0, 0))(x, t)
    influences.extend(tmp_influence)
    if i % 50 == 0:
        print(i)

not_helpful = np.argsort(influences)
helpful = not_helpful[::-1]
