import torch


def init_mu(in_size, batch_size, std=1.0):
    return torch.empty((in_size, batch_size)).normal_(mean=0.0, std=std)


def init_weights(in_size, out_size, std=0.1):
    return torch.empty((in_size, out_size)).normal_(mean=0.0, std=std)


def init_bias(out_size):
    return torch.ones((out_size))


def init_cov(out_size):
    return torch.eye(out_size)


def get_data_sample(rectangle_vector_size, triangle_vector_size, batch_size):
    rectangle_data = torch.ones((rectangle_vector_size, batch_size))
    triangle_data = torch.ones((triangle_vector_size, batch_size))
    return rectangle_data, triangle_data


if __name__ == "__main__":
    """experiment params"""
    num_trials = 100
    num_iters = 100
    batch_size = 32

    """ hyperparams """ ""
    dt = 0.01
    cov_rate = 0.01
    init_mu_std = 1.0
    init_weight_std = 0.1

    """ capsules sizes """
    house_vector_size = 3
    boat_vector_size = 3
    rectangle_vector_size = 3
    triangle_vector_size = 3

    """ house -> rectangle params """
    house_rectangle_weights = init_weights(house_vector_size, rectangle_vector_size, init_weight_std)
    house_rectangle_bias = init_bias(rectangle_vector_size)
    house_rectangle_cov = init_cov(rectangle_vector_size)

    """ house -> triangle params """
    house_triangle_weights = init_weights(house_vector_size, triangle_vector_size, init_weight_std)
    house_triangle_bias = init_bias(triangle_vector_size)
    house_triangle_cov = init_cov(triangle_vector_size)

    """ boat -> rectangle params """
    boat_rectangle_weights = init_weights(boat_vector_size, rectangle_vector_size, init_weight_std)
    boat_rectangle_bias = init_bias(rectangle_vector_size)
    boat_rectangle_cov = init_cov(rectangle_vector_size)

    """ boat -> triangle params """
    boat_triangle_weights = init_weights(boat_vector_size, triangle_vector_size)
    boat_triangle_bias = init_bias(triangle_vector_size)
    boat_triangle_cov = init_cov(triangle_vector_size)

    for trial in range(num_trials):
        print(f"trial {trial}")

        """ variables """
        house_mu = init_mu(house_vector_size, batch_size, init_mu_std)
        boat_mu = init_mu(boat_vector_size, batch_size, init_mu_std)

        """ data """
        rectangle_data, triangle_data = get_data_sample(rectangle_vector_size, triangle_vector_size, batch_size)

        """init predictions"""
        house_rectangle_preds = torch.matmul(house_rectangle_weights, house_mu)
        house_triangle_preds = torch.matmul(house_triangle_weights, house_mu)
        boat_rectangle_preds = torch.matmul(boat_rectangle_weights, boat_mu)
        boat_triangle_preds = torch.matmul(boat_triangle_weights, boat_mu)

        """init errors"""
        house_rectangle_errs = rectangle_data - house_rectangle_preds
        house_triangle_errs = triangle_data - house_triangle_preds
        boat_rectangle_errs = rectangle_data - boat_rectangle_preds
        boat_triangle_errs = triangle_data - boat_triangle_preds

        """ inference """
        for i in range(num_iters):

            """errors"""
            delta = rectangle_data - house_rectangle_preds
            delta = delta - torch.matmul(house_rectangle_cov, house_rectangle_errs)
            house_rectangle_errs = house_rectangle_errs + dt * delta

            delta = triangle_data - house_triangle_preds
            delta = delta - torch.matmul(house_triangle_cov, house_triangle_errs)
            house_triangle_errs = house_triangle_errs + dt * delta

            delta = rectangle_data - boat_rectangle_preds
            delta = delta - torch.matmul(boat_rectangle_cov, boat_rectangle_errs)
            boat_rectangle_errs = boat_rectangle_errs + dt * delta

            delta = triangle_data - boat_triangle_preds
            delta = delta - torch.matmul(boat_triangle_cov, boat_triangle_errs)
            boat_triangle_errs = boat_triangle_errs + dt * delta

            """ update mus """
            delta = torch.matmul(house_rectangle_weights.T, house_mu)
            delta = delta + torch.matmul(house_triangle_weights.T, house_mu)
            house_mu = house_mu + dt * delta

            delta = torch.matmul(boat_rectangle_weights.T, boat_mu)
            delta = delta + torch.matmul(boat_triangle_weights.T, boat_mu)
            house_mu = house_mu + dt * delta

            """ get predictions """
            house_rectangle_preds = torch.matmul(house_rectangle_weights, house_mu)
            house_triangle_preds = torch.matmul(house_triangle_weights, house_mu)
            boat_rectangle_preds = torch.matmul(boat_rectangle_weights, boat_mu)
            boat_triangle_preds = torch.matmul(boat_triangle_weights, boat_mu)

        """ update covariance """
        delta = 0.5 * torch.matmul(house_rectangle_errs, house_rectangle_errs.T)
        delta = delta - torch.linalg.pinv(house_rectangle_cov)
        house_rectangle_cov = house_rectangle_cov + cov_rate * delta

        delta = 0.5 * torch.matmul(house_triangle_errs, house_triangle_errs.T)
        delta = delta - torch.linalg.pinv(house_triangle_cov)
        house_triangle_cov = house_triangle_cov + cov_rate * delta

        delta = 0.5 * torch.matmul(boat_rectangle_errs, boat_rectangle_errs.T)
        delta = delta - torch.linalg.pinv(boat_rectangle_cov)
        house_rectangle_cov = boat_rectangle_cov + cov_rate * delta

        delta = 0.5 * torch.matmul(boat_triangle_errs, boat_triangle_errs.T)
        delta = delta - torch.linalg.pinv(boat_triangle_cov)
        boat_triangle_cov = boat_triangle_cov + cov_rate * delta

        print(house_rectangle_weights)