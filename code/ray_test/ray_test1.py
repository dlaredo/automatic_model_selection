import ray

def main():

	ray.init(redis_address="169.236.181.40:10500")

	x_id = add2.remote(1, 2)
	x = ray.get(x_id)  # 3
	print(x)


@ray.remote
def add2(a, b):
    return a + b


main()