import sys, torch; sys.path.insert(0, "src")
from neuroxnet.inference.aggregate import aggregate_study
def test_mean_logit_then_softmax():
    det = torch.tensor([2.0, 0.0, -2.0, 4.0]); sev = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [3, 0, 0, 0], [0, 0, 0, 1]])
    ph, ps = aggregate_study(det, sev)
    assert torch.isclose(ph, torch.sigmoid(torch.tensor(1.0)))
    assert torch.allclose(ps, torch.softmax(torch.tensor([1.0, 0.25, 0.0, 0.25]), 0))
    assert not torch.isclose(ps[0], torch.softmax(sev, 1).mean(0)[0])   # mean of probabilities is NOT the protocol
