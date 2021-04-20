count = 0
while count < 10:
    print("Im counting")
    time = 0
    while time < 5:
        print("Im waiting")
        if time == 2:
            break
        time += 1
    count += 1