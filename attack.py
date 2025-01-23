import numpy as np


def attack(ranking,history_edges,best_edges):
    best_array = np.array(best_edges)
    best_selected = set(best_edges[len(best_edges) // 2:])
    history_selected = set(history_edges)

    last_ranking = ranking
    last_ranking_array = np.array(last_ranking)
    mal_selected = set(last_ranking[len(last_ranking) // 2:])

    to_promote_set = ((best_selected & history_selected) | (best_selected - mal_selected))
    to_demote_set = ((history_selected - best_selected) | (mal_selected - best_selected))

    to_promote = np.array(list(to_promote_set))
    to_demote = np.array(list(to_demote_set))

    mal_rank = np.empty_like(last_ranking)
    to_demote_count = len(to_demote)
    to_promote_count = len(to_promote)
    to_promote_mask = np.isin(best_array, to_promote)
    a = best_array[to_promote_mask]
    to_demote_mask = np.isin(best_array, to_demote)
    b = best_array[to_demote_mask]

    if to_promote_count > 0:
        mal_rank[-to_promote_count:] = a
    if to_demote_count > 0:
        mal_rank[:to_demote_count] = b

    final_mask = np.isin(best_array, np.concatenate([to_promote, to_demote]))

    # 保持原顺序的其他边
    unchanged = best_array[~final_mask]
    unchanged_count = len(unchanged)
    mal_rank[to_demote_count:to_demote_count+unchanged_count] = unchanged

    x_mask = np.isin(last_ranking_array, mal_rank[to_demote_count:len(mal_rank) //2])
    x = last_ranking_array[x_mask][::-1]
    y_mask = np.isin(last_ranking_array, mal_rank[len(mal_rank) //2:-to_promote_count])
    y = last_ranking_array[y_mask][::-1]
    mal_rank[to_demote_count:len(mal_rank) //2][:len(x)] = x
    mal_rank[len(mal_rank) //2:-to_promote_count][:len(y)] = y

    return mal_rank

def analyze_rank_changes_efficient(current_scores, previous_scores, max_print=5):
    n = len(current_scores)
    threshold = n // 2  # 使用中位数作为阈值

    moved_to_unused = []
    moved_to_used = []
    # current_scores_map = {val: idx for idx, val in enumerate(current_scores)}
    previous_scores_map = {edge: rank for rank, edge in enumerate(previous_scores)}

    for new_rank, edge in enumerate(current_scores):
        old_rank = previous_scores_map[edge]

        if old_rank < threshold <= new_rank:
            moved_to_used.append((edge, old_rank, new_rank))
        elif new_rank < threshold <= old_rank:
            moved_to_unused.append((edge, old_rank, new_rank))

    return moved_to_used, moved_to_unused