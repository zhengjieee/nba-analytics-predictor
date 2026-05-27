"""
The script selects 10 current roster players per team using an activity score that blends 
current-season and previous-season usage so star players are still selected if they are
currently injured or have missed time.

It then downloads all regular season player game logs from the earliest selected debut 
season through the current season.

There are also three additional raw lookup tables:
1. player_info.csv is built from NBA player profile metadata for the selected 300 players, including
height, weight, position, career years, and draft information. 
2. team_stats.csv is built from season-level team totals and win percentages. 
3. opponent_defense.csv is a compact defensive-context lookup derived from team_stats.csv, including
defensive rating, estimated defensive rating, pace, possessions, and win percentage.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers,
    commonplayerinfo,
    commonteamroster,
    leaguedashteamstats,
    leaguegamelog,
)
from nba_api.stats.static import teams


DEFAULT_CURRENT_SEASON = "2025-26"
DEFAULT_PREVIOUS_SEASON = "2024-25"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def generate_seasons(first_year: int, current_season: str) -> list[str]:
    current_start = season_start_year(current_season)
    return [season_label(year) for year in range(first_year, current_start + 1)]


def sleep(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)


def fetch_league_player_logs(season: str, delay: float, timeout: int) -> pd.DataFrame:
    print(f"Downloading league player game log for {season}...")
    logs = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
        timeout=timeout,
    ).get_data_frames()[0]
    sleep(delay)
    logs = logs.rename(columns={"FG3M": "3PM"})
    logs[["FG_PCT", "FG3_PCT", "FT_PCT"]] = logs[["FG_PCT", "FG3_PCT", "FT_PCT"]].fillna("NA")
    logs["SEASON"] = season
    return logs


def fetch_current_rosters(
    current_season: str,
    delay: float,
    timeout: int,
    max_teams: int | None = None,
) -> pd.DataFrame:
    team_rows = teams.get_teams()
    if max_teams:
        team_rows = team_rows[:max_teams]

    rosters = []
    for team in team_rows:
        print(f"Downloading roster for {team['full_name']}...")
        roster = commonteamroster.CommonTeamRoster(
            team_id=team["id"],
            season=current_season,
            timeout=timeout,
        ).get_data_frames()[0]
        sleep(delay)
        roster["TEAM_ID"] = team["id"]
        roster["TEAM_ABBREVIATION"] = team["abbreviation"]
        roster["TEAM_NAME"] = team["full_name"]
        rosters.append(roster)

    return pd.concat(rosters, ignore_index=True)


def activity_by_player(logs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=["PLAYER_ID", f"{prefix}_games", f"{prefix}_minutes"])

    minute_col = "MIN"
    if minute_col not in logs.columns:
        raise ValueError(f"Expected column {minute_col!r} in league game logs.")

    usage = (
        logs.groupby("PLAYER_ID", as_index=False)
        .agg(**{f"{prefix}_games": ("GAME_ID", "nunique"), f"{prefix}_minutes": (minute_col, "sum")})
    )
    return usage


def select_players(
    rosters: pd.DataFrame,
    current_logs: pd.DataFrame,
    previous_logs: pd.DataFrame,
    players_per_team: int,
) -> pd.DataFrame:
    current_usage = activity_by_player(current_logs, "current")
    previous_usage = activity_by_player(previous_logs, "previous")

    selected = (
        rosters.merge(current_usage, on="PLAYER_ID", how="left")
        .merge(previous_usage, on="PLAYER_ID", how="left")
        .fillna(
            {
                "current_games": 0,
                "current_minutes": 0,
                "previous_games": 0,
                "previous_minutes": 0,
            }
        )
    )
    selected["selection_score"] = (
        selected["current_minutes"]
        + 0.7 * selected["previous_minutes"]
        + 10 * selected["current_games"]
        + 7 * selected["previous_games"]
    )

    selected = selected.sort_values(
        ["TEAM_ABBREVIATION", "selection_score", "current_minutes", "previous_minutes"],
        ascending=[True, False, False, False],
    )
    selected = selected.groupby("TEAM_ID", group_keys=False).head(players_per_team)
    return selected.sort_values(["TEAM_ABBREVIATION", "selection_score"], ascending=[True, False])


def add_player_years(selected: pd.DataFrame, current_season: str, delay: float, timeout: int) -> pd.DataFrame:
    print("Downloading player year metadata...")
    all_players = commonallplayers.CommonAllPlayers(
        is_only_current_season=0,
        season=current_season,
        timeout=timeout,
    ).get_data_frames()[0]
    sleep(delay)

    years = all_players[["PERSON_ID", "FROM_YEAR", "TO_YEAR"]].rename(columns={"PERSON_ID": "PLAYER_ID"})
    selected = selected.merge(years, on="PLAYER_ID", how="left")
    selected["FROM_YEAR"] = pd.to_numeric(selected["FROM_YEAR"], errors="coerce")
    selected["TO_YEAR"] = pd.to_numeric(selected["TO_YEAR"], errors="coerce")
    selected.loc[selected["PLAYER_ID"].eq(1642377), "WEIGHT"] = 206
    selected.loc[selected["PLAYER_ID"].eq(1630577), "HOW_ACQUIRED"] = "Signed on 02/16/23"
    return selected


def collect_career_logs(
    selected_players: pd.DataFrame,
    current_season: str,
    delay: float,
    timeout: int,
) -> pd.DataFrame:
    earliest_year = int(selected_players["FROM_YEAR"].min())
    seasons = generate_seasons(earliest_year, current_season)
    selected_ids = set(selected_players["PLAYER_ID"].astype(int))

    logs = []
    for season in seasons:
        season_logs = fetch_league_player_logs(season, delay=delay, timeout=timeout)
        season_logs = season_logs[season_logs["PLAYER_ID"].astype(int).isin(selected_ids)].copy()
        print(f"Keeping {len(season_logs):,} selected-player rows for {season}.")
        logs.append(season_logs)

    return pd.concat(logs, ignore_index=True)


def fetch_player_info(player_ids: list[int], delay: float, timeout: int) -> pd.DataFrame:
    player_info = []
    for index, player_id in enumerate(player_ids, start=1):
        print(f"Downloading player info {index}/{len(player_ids)} for PLAYER_ID {player_id}...")
        info = commonplayerinfo.CommonPlayerInfo(
            player_id=player_id,
            timeout=timeout,
        ).get_data_frames()[0]
        sleep(delay)
        player_info.append(info)

    player_info_df = pd.concat(player_info, ignore_index=True)
    keep_columns = [
        "PERSON_ID",
        "DISPLAY_FIRST_LAST",
        "PLAYER_SLUG",
        "BIRTHDATE",
        "SCHOOL",
        "COUNTRY",
        "HEIGHT",
        "WEIGHT",
        "SEASON_EXP",
        "POSITION",
        "ROSTERSTATUS",
        "TEAM_ID",
        "TEAM_NAME",
        "TEAM_ABBREVIATION",
        "FROM_YEAR",
        "TO_YEAR",
        "DRAFT_YEAR",
        "DRAFT_ROUND",
        "DRAFT_NUMBER",
    ]
    player_info_df = player_info_df[keep_columns].rename(
        columns={
            "PERSON_ID": "PLAYER_ID",
            "DISPLAY_FIRST_LAST": "PLAYER_NAME",
            "DRAFT_NUMBER": "DRAFT_PICK",
        }
    )
    player_info_df.loc[player_info_df["DRAFT_ROUND"].astype(str).eq("0"), "DRAFT_PICK"] = "Undrafted"
    player_info_df.loc[player_info_df["PLAYER_ID"].eq(1642377), "WEIGHT"] = 206
    return player_info_df


def fetch_team_stats(season: str, delay: float, timeout: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading team base stats for {season}...")
    base = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
        timeout=timeout,
    ).get_data_frames()[0]
    sleep(delay)

    print(f"Downloading team advanced stats for {season}...")
    advanced = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
        timeout=timeout,
    ).get_data_frames()[0]
    sleep(delay)

    base = base.rename(columns={"FG3M": "3PM"})
    base["SEASON"] = season
    advanced["SEASON"] = season
    return base, advanced


def collect_supporting_data(
    selected_players: pd.DataFrame,
    seasons: list[str],
    delay: float,
    timeout: int,
) -> None:
    player_ids = selected_players["PLAYER_ID"].astype(int).drop_duplicates().tolist()
    player_info = fetch_player_info(player_ids, delay=delay, timeout=timeout)
    player_info_path = RAW_DIR / "player_info.csv"
    player_info.to_csv(player_info_path, index=False)
    print(f"Saved {len(player_info):,} player info rows to {player_info_path}.")

    base_stats = []
    advanced_stats = []
    for season in seasons:
        base, advanced = fetch_team_stats(season, delay=delay, timeout=timeout)
        base_stats.append(base)
        advanced_stats.append(advanced)

    team_stats = pd.concat(base_stats, ignore_index=True)
    advanced_team_stats = pd.concat(advanced_stats, ignore_index=True)
    team_stats_path = RAW_DIR / "team_stats.csv"
    team_stats.to_csv(team_stats_path, index=False)
    print(f"Saved {len(team_stats):,} team stat rows to {team_stats_path}.")

    opponent_defense_columns = [
        "SEASON",
        "TEAM_ID",
        "TEAM_NAME",
        "GP",
        "W",
        "L",
        "W_PCT",
        "E_DEF_RATING",
        "DEF_RATING",
        "DEF_RATING_RANK",
        "PACE",
        "POSS",
    ]
    opponent_defense = advanced_team_stats[opponent_defense_columns].copy()
    opponent_defense_path = RAW_DIR / "opponent_defense.csv"
    opponent_defense.to_csv(opponent_defense_path, index=False)
    print(f"Saved {len(opponent_defense):,} opponent defense rows to {opponent_defense_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NBA career game logs for selected current players.")
    parser.add_argument("--current-season", default=DEFAULT_CURRENT_SEASON)
    parser.add_argument("--previous-season", default=DEFAULT_PREVIOUS_SEASON)
    parser.add_argument("--players-per-team", type=int, default=10)
    parser.add_argument("--max-teams", type=int, default=None, help="Use a small number for smoke tests.")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between NBA API calls in seconds.")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--selection-only", action="store_true", help="Only save selected_players.csv.")
    parser.add_argument(
        "--supporting-only",
        action="store_true",
        help="Download player_info.csv, team_stats.csv, and opponent_defense.csv for existing raw logs.",
    )
    parser.add_argument(
        "--use-existing-selection",
        action="store_true",
        help="Reuse data/raw/selected_players.csv instead of downloading rosters and recalculating selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = RAW_DIR / "selected_players.csv"

    if args.use_existing_selection:
        selected = pd.read_csv(selected_path)
        print(f"Loaded {len(selected):,} selected players from {selected_path}.")
    else:
        rosters = fetch_current_rosters(
            current_season=args.current_season,
            delay=args.delay,
            timeout=args.timeout,
            max_teams=args.max_teams,
        )
        current_logs = fetch_league_player_logs(args.current_season, delay=args.delay, timeout=args.timeout)
        previous_logs = fetch_league_player_logs(args.previous_season, delay=args.delay, timeout=args.timeout)

        selected = select_players(
            rosters=rosters,
            current_logs=current_logs,
            previous_logs=previous_logs,
            players_per_team=args.players_per_team,
        )
        selected = add_player_years(selected, args.current_season, delay=args.delay, timeout=args.timeout)

        selected.to_csv(selected_path, index=False)
        print(f"Saved {len(selected):,} selected players to {selected_path}.")

    if args.selection_only:
        return

    if args.supporting_only:
        logs_path = RAW_DIR / "player_game_logs.csv"
        logs = pd.read_csv(logs_path, usecols=["SEASON"])
        seasons = sorted(logs["SEASON"].dropna().unique(), key=season_start_year)
        collect_supporting_data(
            selected_players=selected,
            seasons=seasons,
            delay=args.delay,
            timeout=args.timeout,
        )
        return

    logs = collect_career_logs(
        selected_players=selected,
        current_season=args.current_season,
        delay=args.delay,
        timeout=args.timeout,
    )
    logs_path = RAW_DIR / "player_game_logs.csv"
    logs.to_csv(logs_path, index=False)
    print(f"Saved {len(logs):,} player-game rows to {logs_path}.")


if __name__ == "__main__":
    main()
