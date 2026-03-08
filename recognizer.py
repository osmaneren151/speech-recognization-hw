import numpy as np
from hmmlearn import hmm

OBS_MAP = {
    "High": 0,
    "Low": 1,
    "Mid": 2
}

def encode_observations(obs_sequence):
    return np.array([[OBS_MAP[o]] for o in obs_sequence], dtype=int)

def create_categorical_hmm(startprob, transmat, emissionprob):
    model = hmm.CategoricalHMM(n_components=len(startprob), init_params="")
    model.startprob_ = np.array(startprob)
    model.transmat_ = np.array(transmat)
    model.emissionprob_ = np.array(emissionprob)
    return model

ev_startprob = [1.0, 0.0]
ev_transmat = [
    [0.6, 0.4],
    [0.2, 0.8]
]
ev_emissionprob = [
    [0.7, 0.3, 0.0],
    [0.1, 0.9, 0.0]
]
ev_model = create_categorical_hmm(ev_startprob, ev_transmat, ev_emissionprob)

okul_startprob = [1.0, 0.0, 0.0, 0.0]
okul_transmat = [
    [0.6, 0.4, 0.0, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.0, 0.6, 0.4],
    [0.0, 0.0, 0.0, 1.0]
]
okul_emissionprob = [
    [0.6, 0.2, 0.2],
    [0.2, 0.3, 0.5],
    [0.3, 0.2, 0.5],
    [0.1, 0.8, 0.1]
]
okul_model = create_categorical_hmm(okul_startprob, okul_transmat, okul_emissionprob)

def classify_observation_sequence(obs_sequence):
    X = encode_observations(obs_sequence)
    ev_score = ev_model.score(X)
    okul_score = okul_model.score(X)
    predicted_word = "EV" if ev_score > okul_score else "OKUL"
    return predicted_word, ev_score, okul_score

def decode_ev_viterbi(obs_sequence):
    X = encode_observations(obs_sequence)
    logprob, states = ev_model.decode(X, algorithm="viterbi")
    state_names = ["e", "v"]
    decoded_states = [state_names[s] for s in states]
    return logprob, decoded_states

def main():
    print("=== Speech Recognition Mini Project ===")
    print()

    test_sequences = [
        ["High", "Low"],
        ["Mid", "Mid", "Low", "Low"],
        ["High", "Low", "Low"],
        ["Low", "Low"]
    ]

    for seq in test_sequences:
        predicted, ev_score, okul_score = classify_observation_sequence(seq)

        print(f"Observation sequence: {seq}")
        print(f"EV score   : {ev_score:.4f}")
        print(f"OKUL score : {okul_score:.4f}")
        print(f"Prediction : {predicted}")

        if seq == ["High", "Low"]:
            logprob, decoded = decode_ev_viterbi(seq)
            print(f"EV Viterbi states : {decoded}")
            print(f"Viterbi logprob   : {logprob:.4f}")

        print("-" * 45)

if __name__ == "__main__":
    main()
