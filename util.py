import torch


def hamilton_product(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return torch.tensor(
        [
            a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
        ]
    )


def validate_input(Q):
    assert Q.dim() == 2 and Q.shape[1] == 4, "The shape of quaternions must be B x 4"
