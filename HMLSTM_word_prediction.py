from hmlstm import HMLSTMNetwork, prepare_inputs, get_text, viz_char_boundaries

print('load and prepare input')

batch_size = 64
num_batches = 50
truncate_len = 512
step_size = truncate_len//2
# text_path = 'sample.txt'
text_path = 'text8.txt'

batches_in, batches_out = prepare_inputs(batch_size=batch_size, truncate_len=truncate_len,
                                         step_size=step_size, text_path=text_path, num_batches=num_batches)

print('creating network')
output_size = 27
input_size = 27
# embed_size = 2048
# out_hidden_size = 1024
# hidden_state_sizes = 1024
embed_size = 2048
out_hidden_size = 1024
hidden_state_sizes = 1024
task = 'classification'

network = HMLSTMNetwork(output_size=output_size, input_size=input_size, embed_size=embed_size,
                        out_hidden_size=out_hidden_size, hidden_state_sizes=hidden_state_sizes,
                        task=task)

print('train network')
n_epochs = 5
network.train(batches_in[:-1], batches_out[:-1], save_vars_to_disk=True,
              load_vars_from_disk=False, variable_path='./text8_ckpt', epochs=n_epochs)

predictions = network.predict(batches_in[-1], variable_path='./text8_ckpt')
boundaries = network.predict_boundaries(batches_in[-1], variable_path='./text8_ckpt')

# visualize boundaries
viz_char_boundaries(get_text(batches_out[-1][0]), get_text(predictions[0]), boundaries[0])


