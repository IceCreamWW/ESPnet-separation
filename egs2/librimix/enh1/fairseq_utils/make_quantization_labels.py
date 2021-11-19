import torch
import fairseq
import argparse
import torchaudio
from tqdm import tqdm
import pdb

def load_model(path_to_model):
    cp = torch.load(path_to_model)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
    model = model[0]
    model.eval()
    return model

def get_idx(model, path_to_wav):
    audio_input = torchaudio.load(path_to_wav)
    z = model.feature_extractor(wav_input_16khz)
    _, idxs = model.vector_quantizer.forward_idx(z)
    return idxs.squeeze()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make quantization labels for audios with vq-wav2vec model')
    parser.add_argument('--model', type=str, help='path to vq-wav2vec model')
    parser.add_argument('--wav_scp', type=str, help='path to wav.scp')
    parser.add_argument('--outdir', type=str, help='path output directory')

    model = load_model(args.model)
    groups = model.vector_quantizer.groups
    out_fps = [open(os.path.join(args.outdir, i), 'w') for i in range(groups)]

    num_lines = sum(1 for line in open(args.wav_scp))
    with open(args.wav_scp) as fp:
        for line in tqdm(f, total=num_lines):
            uttid, path = line.strip().split()
            idxs = get_idx(model, path)

    for fp in out_fps:
        fp.close()

