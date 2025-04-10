import csv
import heapq
import math
import time
import random
from datetime import datetime, timedelta


class Node:
    def __init__(self, name, lat, lon):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def __repr__(self):
        return str(self.name) + ', ' + str(self.lat) + ', ' + str(self)


class Connection:
    def __init__(self, start_node: Node, end_node: Node, departure_time, arrival_time, line):
        self.start_node = start_node
        self.end_node = end_node
        self.departure_time = parse_time(departure_time)
        self.arrival_time = parse_time(arrival_time)
        self.minutes_time_diff = time_difference_minutes(departure_time, arrival_time)
        self.meters_diff = haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon)
        self.line = line

    def __repr__(self):
        return str(self.start_node) + ', ' + str(self.end_node)

    def __lt__(self, other):
        return self.minutes_time_diff < other.minutes_time_diff


def time_difference_minutes(time1, time2):
    if not isinstance(time1, datetime):
        time1 = parse_time(time1)
    if not isinstance(time2, datetime):
        time2 = parse_time(time2)
    return (time2 - time1).total_seconds() / 60


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    diff_lon = abs(lon2 - lon1)
    diff_lat = abs(lat2 - lat1)
    a = math.sin(diff_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(diff_lon / 2) ** 2
    distance_meters = 2 * math.asin(math.sqrt(a)) * 6371000

    return distance_meters


def parse_time(time_str):
    hour = int(time_str[:2])
    if hour >= 24:
        return datetime.strptime(str(hour - 24) + time_str[2:], "%H:%M:%S") + timedelta(days=1)
    else:
        return datetime.strptime(time_str, "%H:%M:%S")


def load_and_build_graph():
    stop_name_to_node = {}

    with open('connection_graph.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            start_name = row['start_stop']
            end_name = row['end_stop']

            if start_name not in stop_name_to_node:
                stop_name_to_node[start_name] = Node(
                    start_name,
                    float(row['start_stop_lat']),
                    float(row['start_stop_lon'])
                )

            if end_name not in stop_name_to_node:
                stop_name_to_node[end_name] = Node(
                    end_name,
                    float(row['end_stop_lat']),
                    float(row['end_stop_lon'])
                )

            start_node = stop_name_to_node[start_name]
            end_node = stop_name_to_node[end_name]

            connection = Connection(
                start_node=start_node,
                end_node=end_node,
                departure_time=row['departure_time'],
                arrival_time=row['arrival_time'],
                line=row['line']
            )

            start_node.add_connection(connection)

    return stop_name_to_node


# --- ALGORITHMS ---
def dijkstra(graph, start_name: str, end_name: str, start_time: datetime):
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    pq = [(0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    parent = {}

    while pq:
        current_time_cost, current_line, current_time, prev_connection, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() // 60) + connection.minutes_time_diff + current_time_cost

                if next_node.name not in time_costs or new_cost < time_costs[next_node.name]:
                    time_costs[next_node.name] = new_cost
                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

                    heapq.heappush(pq, (new_cost, connection.line, connection.arrival_time, connection, next_node))

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def a_star(graph, start_name: str, end_name: str, start_time: datetime):
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    def heuristic(start_node, end_node):
        return haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon) / 10000

    pq = [(0 + heuristic(start_node, end_node), 0, 0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    parent = {}

    while pq:
        heuristic_cost, current_time_cost, transfers, current_line, current_time, prev_connection, current_node = heapq.heappop(
            pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() // 60) + time_difference_minutes(connection.departure_time,
                                                                                          connection.arrival_time) + current_time_cost

                if next_node.name not in time_costs or new_cost < time_costs[next_node.name]:
                    time_costs[next_node.name] = new_cost
                    new_heuristic_cost = new_cost + heuristic(next_node, end_node)
                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

                    new_transfers = transfers
                    if current_line is not None and connection.line != current_line:
                        new_transfers = transfers + 1

                    heapq.heappush(pq, (
                        new_heuristic_cost, new_cost, new_transfers, connection.line, connection.arrival_time,
                        connection,
                        next_node))

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def a_star_improved(graph, start_name: str, end_name: str, start_time: datetime):
    # amplifier = float(input("Enter the heuristic amplifier: "))
    amplifier = 4
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    def heuristic(start_node, end_node):
        return haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon) / 1000 * amplifier

    pq = [(0 + heuristic(start_node, end_node), 0, 0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    parent = {}

    while pq:
        heuristic_cost, current_time_cost, transfers, current_line, current_time, prev_connection, current_node = heapq.heappop(
            pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() / 60) + connection.minutes_time_diff + current_time_cost
                new_heuristic_cost = new_cost + heuristic(next_node, end_node)

                new_transfers = transfers
                if current_line is not None or connection.line != current_line:
                    new_transfers += 1

                if next_node.name not in time_costs or new_cost < time_costs[next_node.name]:
                    time_costs[next_node.name] = new_cost

                    heapq.heappush(pq, (
                        new_heuristic_cost, new_cost, new_transfers, connection.line, connection.arrival_time,
                        connection,
                        next_node))

                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def a_star_transfers(graph, start_name: str, end_name: str, start_time: datetime):
    # amplifier = float(input("Enter the heuristic amplifier: "))
    amplifier = 4
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    def heuristic(start_node, end_node):
        return haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon) / 1000 * amplifier

    pq = [(0, 0 + heuristic(start_node, end_node), 0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    transfer_costs = {start_name: 0}
    parent = {}

    while pq:
        transfers, heuristic_cost, current_time_cost, current_line, current_time, prev_connection, current_node = heapq.heappop(
            pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() // 60) + connection.minutes_time_diff + current_time_cost

                new_transfers = transfers
                if current_line is not None and connection.line != current_line:
                    new_transfers = transfers + 1

                if (next_node.name not in transfer_costs or new_transfers < transfer_costs[next_node.name] or
                        (new_transfers == transfer_costs[next_node.name] and new_cost < time_costs[next_node.name])):
                    time_costs[next_node.name] = new_cost
                    transfer_costs[next_node.name] = new_transfers

                    new_heuristic_cost = new_cost + heuristic(next_node, end_node)

                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

                    heapq.heappush(pq, (
                        new_transfers, new_heuristic_cost, new_cost, connection.line, connection.arrival_time,
                        connection,
                        next_node))

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def print_stats(path, start_name, cost, calc_time, end_name, stops=None):
    if path is None:
        return

    print("Starting from " + start_name)

    print("  Line\t | Departure\t | Arrival\t | Stops")
    for current_node, next_node in zip(path, path[1:]):
        print(f"\t{current_node[1]}\t"
              f" | {datetime.strftime(current_node[2], "%H:%M:%S")}\t\t"
              f" | {datetime.strftime(current_node[3], "%H:%M:%S")}\t"
              f" | {current_node[0]} -> {next_node[0] if next_node is not None else end_name}")

        if stops is not None and next_node[0] in stops:
            print("Reached stop " + next_node[0])

        if next_node[1] is not None and current_node[1] != next_node[1]:
            print("Warning! Line change")

    print(f"Total time cost: \t\t\t{cost} minutes")
    print(f"Total calculation time: \t{calc_time * 1000} ms")


def tabu_search(graph, start_stop, stops_list, start_time, optimization_criterion):
    def solve_segment(from_stop, to_stop, departure_time):
        if optimization_criterion == 't':
            segment, seg_cost, calc_time = a_star_improved(graph, from_stop, to_stop, departure_time)
        elif optimization_criterion == 'p':
            segment, seg_cost, calc_time = a_star_transfers(graph, from_stop, to_stop, departure_time)
        else:
            print("Wrong optimization criterion")
            return None, None, None
        if segment is None:
            return None, None, None

        arrival_time = segment[-2][3] + timedelta(minutes=5) if segment[-2][3] is not None else departure_time
        return segment, seg_cost + 5, arrival_time

    def evaluate_route(perm):
        total_cost = 0
        full_path = []
        current_time = start_time

        route = [start_stop] + perm + [start_stop]
        for i in range(len(route) - 1):
            from_stop = route[i]
            to_stop = route[i + 1]
            segment, seg_cost, arrival_time = solve_segment(from_stop, to_stop, current_time)

            if segment is None:
                return float('inf'), None

            if i == len(route) - 2:
                full_path.extend(segment)
            else:
                full_path.extend(segment[:-1])

            total_cost += seg_cost
            current_time = arrival_time
        return total_cost, full_path

    start_time_calc = time.time()

    current_perm = stops_list[:]
    best_cost, best_full_path = evaluate_route(current_perm)

    tabu_tenure = min(10, len(stops_list) * (len(stops_list) - 1) // 2)
    tabu_list = {}

    max_iterations = 50
    no_improvement = 0

    for iteration in range(max_iterations):
        candidate_found = False
        best_candidate_cost = float('inf')
        best_candidate_perm = None
        best_candidate_move = None
        best_candidate_path = None

        # sampling
        n = len(current_perm)
        all_possible_moves = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        sample_size = min(n * (n - 1) // 2, 10)
        sampled_moves = random.sample(all_possible_moves, sample_size)

        for i, j in sampled_moves:
            candidate_perm = current_perm[:]
            candidate_perm[i], candidate_perm[j] = candidate_perm[j], candidate_perm[i]
            move = (i, j)

            candidate_cost, candidate_path = evaluate_route(candidate_perm)

            if (move in tabu_list or (j, i) in tabu_list) and candidate_cost >= best_cost:  # aspiration
                continue

            if candidate_cost < best_candidate_cost:
                best_candidate_cost = candidate_cost
                best_candidate_perm = candidate_perm
                best_candidate_move = move
                best_candidate_path = candidate_path
                candidate_found = True

        if not candidate_found:
            break

        current_perm = best_candidate_perm

        tabu_list[best_candidate_move] = iteration + tabu_tenure
        tabu_list = {
            moves: move_iteration
            for moves, move_iteration in tabu_list.items()
            if move_iteration > iteration
        }

        if best_candidate_cost < best_cost:
            best_cost = best_candidate_cost
            best_full_path = best_candidate_path
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= 20:
            break

    return best_full_path, best_cost, time.time() - start_time_calc


def main():
    start_time_calc = time.time()
    graph = load_and_build_graph()
    print(f"Graph loaded in {time.time() - start_time_calc} seconds")

    print("\n--- Jakdojade ---\n")
    while True:
        task = input("Enter task to start 1/2 (or 'q' to quit): ")

        if task == 'q':
            print("Goodbye!")
            break

        if task == '1':
            start_node = input("Enter start node: ")
            end_node = input("Enter end node: ")
            start_time = input("Enter start time HH:MM:SS: ")
            optimization_criterion = input("Enter optimization p/t: ")

            if optimization_criterion == 't':
                path, cost, calc_time = a_star_improved(graph, start_node, end_node, parse_time(start_time))
                _, d_cost, d_calc_time = dijkstra(graph, start_node, end_node, parse_time(start_time))
                _, a_cost, a_calc_time = a_star(graph, start_node, end_node, parse_time(start_time))
                print_stats(path, start_node, cost, calc_time, end_node)

                print("\n--- Comparison with usual A-star ---")
                print(f"A-star time cost: \t\t\t{a_cost} minutes")
                print(f"A-star calculation time: \t{a_calc_time * 1000} ms")

                print("--- Comparison with Dijkstra ---")
                print(f"Dijkstra time cost: \t\t{d_cost} minutes")
                print(f"Dijkstra calculation time: \t{d_calc_time * 1000} ms\n")

            elif optimization_criterion == 'p':
                path, cost, calc_time = a_star_transfers(graph, start_node, end_node, parse_time(start_time))
                print_stats(path, start_node, cost, calc_time, end_node)
            else:
                print("Wrong optimization criteria")
        if task == '2':
            start_node = input("Enter start node: ")
            to_visit_input = input("Enter list of stops to visit (semicolon-separated): ")
            start_time = input("Enter start time HH:MM:SS: ")
            optimization_criterion = input("Enter optimization t/p: ")

            to_visit = to_visit_input.split(";")

            path, cost, calc_time = tabu_search(graph, start_node, to_visit, parse_time(start_time),
                                                optimization_criterion)

            if path is None:
                print("No valid route found")
            else:
                print_stats(path, start_node, cost, calc_time, start_node, to_visit)


if __name__ == "__main__":
    main()
"""
GRABISZYŃSKA (Cmentarz)
Krakowska
15:05:00

Psie Pole (Rondo Lotników Polskich)
FAT
12:55:00

FAT
Bałtycka;pl. Bema;PL. GRUNWALDZKI
12:55:00
"""
