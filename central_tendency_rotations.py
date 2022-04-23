import torch

from util import hamilton_product, validate_input

__version__ = "1.0"


def mean(Q, weights=None):
    if weights is None:
        weights = torch.ones(len(Q)) / len(Q)
    validate_input(Q)
    A = torch.zeros((4, 4), device=torch.device("cuda:0"))
    weight_sum = torch.sum(weights)

    oriented_Q = ((Q[:, 0:1] > 0).float() - 0.5) * 2 * Q
    A = torch.einsum("bi,bk->bik", (oriented_Q, oriented_Q))
    A = torch.sum(torch.einsum("bij,b->bij", (A, weights)), 0)
    A /= weight_sum

    q_avg = torch.linalg.eigh(A)[1][:, -1]
    if q_avg[0] < 0:
        return -q_avg
    return q_avg


def median(Q, p=1, max_angular_update=0.0001, max_iterations=1000):
    validate_input(Q)
    weights = torch.ones(len(Q)) / len(Q)
    q_median = mean(Q, weights)
    Q = Q.cpu()  # because we do sequential operations
    q_median = q_median.cpu()
    EPS_ANGLE = 0.0000001
    max_angular_update = max(max_angular_update, EPS_ANGLE)
    theta = 10 * max_angular_update
    i = 0

    while theta > max_angular_update and i <= max_iterations:
        delta = torch.zeros(3)
        weight_sum = 0
        for q in Q:
            qj = hamilton_product(
                q, torch.tensor([q_median[0], -q_median[1], -q_median[2], -q_median[3]])
            )
            theta = 2 * torch.acos(qj[0])
            if theta > EPS_ANGLE:
                axis_angle = qj[1:] / torch.sin(theta / 2)
                axis_angle *= theta
                weight = 1.0 / pow(theta, 2 - p)
                delta += weight * axis_angle
                weight_sum += weight
        if weight_sum > EPS_ANGLE:
            delta /= weight_sum
            theta = torch.linalg.norm(delta)
            if theta > EPS_ANGLE:
                stby2 = torch.sin(theta * 0.5)
                delta /= theta
                q = torch.tensor(
                    [
                        torch.cos(theta * 0.5),
                        stby2 * delta[0],
                        stby2 * delta[1],
                        stby2 * delta[2],
                    ]
                )
                q_median = hamilton_product(q, q_median)
                if q_median[0] < 0:
                    q_median *= -1
        else:
            theta = 0
            i += 1
    return q_median
