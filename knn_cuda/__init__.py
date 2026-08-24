import torch


class KNN:

    def __init__(
        self,
        k=1,
        transpose_mode=False
    ):
        self.k = k
        self.transpose_mode = transpose_mode


    def __call__(self, ref, query):

        """
        Compatible with KNN_CUDA

        ref:
            (B, N, 3)

        query:
            (B, M, 3)

        return:
            dist:
                (B, M, k)

            idx:
                (B, M, k)
        """

        if self.transpose_mode:
            # KNN_CUDA transpose_mode=True
            # expects:
            # (B,3,N)
            #
            # converts to:
            # (B,N,3)

            if ref.shape[1] == 3:
                ref = ref.transpose(1,2)

            if query.shape[1] == 3:
                query = query.transpose(1,2)


        # pairwise distance
        dist = torch.cdist(
            query,
            ref,
            p=2
        )


        dist_k, idx = torch.topk(
            dist,
            self.k,
            dim=-1,
            largest=False,
            sorted=True
        )


        return dist_k, idx