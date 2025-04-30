import torch

class Polar(torch.nn.Module):
    def __init__(self, output_shape=None):
        super().__init__()
        self.output_shape=output_shape

    def forward(self, data):
        input_shape_x, input_shape_y = data.shape[1], data.shape[0]
        center_x, center_y = input_shape_x / 2, input_shape_y/2
        MAX_R = torch.tensor(data.shape).float().norm() / 2

        theta, r = torch.meshgrid(torch.arange(self.output_shape[0]).type_as(data), torch.arange(self.output_shape[1]).type_as(data), indexing='ij')
        theta = theta.float()
        r = r.float()
        X = center_x + (r * MAX_R / self.output_shape[1]) * torch.cos(theta * 2 * torch.pi / self.output_shape[0])
        Y = center_y - (r * MAX_R / self.output_shape[1]) * torch.sin(theta * 2 * torch.pi / self.output_shape[0])

        mask = (0 <= X) & (X < input_shape_x) & (0 <= Y) & (Y < input_shape_y)
        return mask * data[:, Y.long().clamp(0, input_shape_y-1),X.long().clamp(0, input_shape_x - 1)]

if __name__ == '__main__':
    from skimage.transform import warp_polar

    data = torch.arange(3*180*180).reshape(3,180,180).float()
    output_shape = (190,165)

    import timeit
    N = 1000

    polar = Polar(output_shape)

    reference = timeit.timeit(lambda: torch.from_numpy(warp_polar(data.numpy(),scaling='log', output_shape=output_shape, channel_axis=0)), number=N)
    print('ref', reference)
    implemented = timeit.timeit(lambda: polar(data), number=N)
    print('implemented', implemented)

    data = data.cuda()
    print('CUDA')

    reference = timeit.timeit(lambda: torch.from_numpy(warp_polar(data.cpu().numpy(),scaling='log', output_shape=output_shape, channel_axis=0)), number=N)
    print('ref', reference)
    implemented = timeit.timeit(lambda: polar(data), number=N)
    print('log polar', implemented)

