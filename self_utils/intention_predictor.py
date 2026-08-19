import sys
import os
import torch


class IntentionPredictor:
    """
    Wraps the PedAST-GCN spatial-temporal GCN to predict pedestrian
    crossing intention from a 16-frame sliding window.

    Expected checkpoint format:  {"model": state_dict, "epoch": ..., "acc": ...}
    """

    def __init__(self, model_path, device):
        """
        Args:
            model_path (str): path to best.pth weight file
            device: GPU index string (e.g. '0') or 'cpu', or torch.device
        """
        if isinstance(device, str):
            self.device = torch.device('cpu') if device == 'cpu' else torch.device('cuda', int(device))
        else:
            self.device = device

        # ── temporarily add PedAST-GCN/ to sys.path so that st_gcn.py
        #    can resolve its own imports (utils.tgcn, graph) ──────────────
        #
        # Problem: the project root's `utils/` package is already cached in
        # sys.modules, so Python would find it first instead of
        # PedAST-GCN/utils/tgcn.py.  We temporarily evict the conflicting
        # entries, import st_gcn, then restore them.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pedast_dir   = os.path.join(project_root, 'PedAST-GCN')

        # Keys that belong to the main project's utils and must be restored
        _conflict_keys = [k for k in sys.modules if k == 'utils' or k.startswith('utils.')]
        _saved_modules  = {k: sys.modules.pop(k) for k in _conflict_keys}

        sys.path.insert(0, pedast_dir)
        try:
            from st_gcn import Model
        finally:
            # Restore sys.path
            if sys.path and sys.path[0] == pedast_dir:
                sys.path.pop(0)
            # Remove PedAST-GCN's utils.* from the cache so they don't
            # shadow the project's utils on later imports
            for k in list(sys.modules):
                if (k == 'utils' or k.startswith('utils.')) and k not in _saved_modules:
                    sys.modules.pop(k, None)
            # Restore the project's original utils modules
            sys.modules.update(_saved_modules)

        # ── build model ───────────────────────────────────────────────────
        # in_channels_skeleton=3  (x_rel, y_rel, conf)
        # in_channels_box=2       (x/img_w, y/img_h)
        # num_class=1             (binary sigmoid output)
        self.model = Model(
            in_channels_skeleton=3,
            in_channels_box=2,
            num_class=1,
            edge_importance_weighting_skeleton=True,
            edge_importance_weighting_box=False,
        )

        # ── load weights ──────────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, skeleton_t, box_t):
        """
        Run crossing-intention inference (supports batch).

        Args:
            skeleton_t (Tensor): (B, 16, 18, 3)  B=1 or more
            box_t      (Tensor): (B, 16,  4, 2)

        Returns:
            B=1 : (label_str, prob_float)
            B>1 : (list[label_str], list[prob_float])
        """
        skeleton_t = skeleton_t.float().to(self.device)
        box_t      = box_t.float().to(self.device)

        with torch.no_grad():
            out = self.model(skeleton_t, box_t).view(-1)   # (B,)

        probs  = out.cpu().tolist()
        labels = ['crossing' if p >= 0.5 else 'not crossing' for p in probs]
        return labels, probs
