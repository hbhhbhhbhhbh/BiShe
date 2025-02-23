class CombinedLoss(nn.Module):
    def __init__(self, idc, surface_loss_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.idc = idc
        self.surface_loss_weight = surface_loss_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.surface_loss = SurfaceLoss(idc=idc)

    def forward(self, logits, edge_logits, targets, dist_maps):
        ce_loss = self.ce_loss(logits, targets)
        surface_loss = self.surface_loss(edge_logits, dist_maps)
        total_loss = ce_loss + self.surface_loss_weight * surface_loss
        return total_loss