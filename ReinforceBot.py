import math
import os
import sys
import time
import pickle
import random
from typing import List, Tuple, Dict, Optional, Any


import cv2
import numpy as np

from sc2.bot_ai import BotAI
from sc2.main import run_game
from sc2 import maps, position
from sc2.player import Bot, Computer
from sc2.position import Point2, Point3
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.data import Difficulty, Race, Result

# disable oneDNN TF optimizations (keeps behavior consistent)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



class Lemonek(BotAI):

    def __init__(self) -> None:
        self.replays_enabled: bool = True

        self.times: list = []
        self.mineral_amounts: list = []

        self.map_width, self.map_height = 0, 0
        self.old_mineral_amount: int = 0
        self.new_mineral_amount: int = 0
        self.sum_minerals: int = 0
        self.last_scout_iteration = 0
        self.ticks_to_scout = 180

        self.reward: float = 0.0
        self.game_map: Optional[np.ndarray] = None
        self.last_tick: int = 0

    def _reward_agent(self, iteration: int) -> float:
        """Update and return the accumulated reward for the agent."""
        self.reward = 0
        try:
            for marine in self.units(UnitTypeId.MARINE):
                if marine.is_attacking and  marine.target_in_range:
                    if (
                        self.enemy_units.closer_than(10, marine) 
                        or
                        self.enemy_structures.closer_than(10, marine)
                        ): 
                        self.reward += 0.02
        # try:
        #     for unit in self.units(UnitTypeId.SUPPLYDEPOT):
        #         if unit == self.units(UnitTypeId.SUPPLYDEPOT).not_ready:
        #             nagroda += 0.01
                

        # except Exception as problem:
        #     nagroda = 0
        #     print(problem)

        # try:
        #     for unit in self.units(UnitTypeId.BARRACKS):
        #         if unit == self.units(UnitTypeId.BARRACKS).not_ready:
        #             nagroda += 0.03
                

        # except Exception as problem:
        #     nagroda = 0
        #     print(problem)

        except Exception as e: 
            self.reward = 0.0; print(e)

        if iteration % 100 == 0:
            print(f"Iteracja: {iteration} | Aktualna nagroda: {self.reward}| "
                  f"Ilosc marine: {self.units(UnitTypeId.MARINE).amount}")


    def _save_score(self) -> None:
        """Save a small game-state dict to disk."""
        game_state_dict: Dict[str, Any] = {"game_map": self.game_map
                                           , "reward": self.reward
                                           , "moves": None
                                           , "game_finished": False
                                           }
        try:
            with open("sarsa.pkl", "wb") as f:
                pickle.dump(game_state_dict, f)
        except Exception as e:
            print(f"Nie udalo sie zapisac wyniku gry - {e}")

    def _random_position_variance(self, enemy_start_pos: tuple) -> Point2:
        """Return a random point close to enemy starting position.

        Original logic used a random direction (0, 4, or center). This preserves
        the behavior but uses clearer naming and bounds checking.
        """
        direction = random.randint(0, 4)
        if direction == 0:
            x = enemy_start_pos[0] - 40
            y = enemy_start_pos[1] - 15
        elif direction == 4:
            x = enemy_start_pos[0] + 30
            y = enemy_start_pos[1] + 10
        else:
            x = enemy_start_pos[0]
            y = enemy_start_pos[1]


        
        x = 0 if x < 0 else x
        x = self.map_width if x > self.map_width else x

        y = 0 if y < 0 else y
        y = self.map_height if y > self.map_height else y

        goto = position.Point2(position.Pointlike((x, y)))
        return goto

    # Do przyszlego uzycia?
    # def assign_medivac(self):
    #     for med in self.units(UnitTypeId.MEDIVAC).idle:
    #         med.move(random.choice(self.units(UnitTypeId.MARINE)))

    async def on_end(self, game_result: Result) -> None:
        """Writes a small timing log on game end."""
        print("-" * 20)
        print("||                            ||")
        print("||*** on_end method called ***||")
        print("||                            ||")
        print("-" * 20)

        try:
            with open("czasiwynik.txt", "a") as f:
                f.write(
                    "Model {} - time {} \n".format(game_result, self.state.game_loop / 22.4)
                )
        except Exception as e:
            print(f"Nie udalo sie zapisac czasu i wyniku w pliku tekstowym - {e}")

    async def on_step(self, iteration: int) -> None:

        """Main on-step logic. Loads last saved SARSA state and executes actions."""
        self.map_width, self.map_height = my_bot.game_info.map_size[0], my_bot.game_info.map_size[1]
        moves_available = False
        sarsa_state = None

        # Wait until a valid sarsa.pkl (game info dictionary) is available
        while not moves_available:
            try:
                with open("sarsa.pkl", "rb") as f:
                    sarsa_state = pickle.load(f)

                if sarsa_state is None or sarsa_state["moves"] is None: 
                    moves_available = False
                else:
                    moves_available = True
            except:
                pass
            # except Exception:
            #     await asyncio.sleep(0) # yields control to other tasks; program doesn't freeze
            #     continue

        if iteration % 50 == 0:
            await self.distribute_workers()

        move = sarsa_state["moves"]
        if iteration % 5 == 0:
            if move == 0:
                await self.expand()
            elif move == 1:
                await self.build_barracks()
            elif move == 2:
                await self.train_marine()
            elif move == 3:
                await self.train_scv()
            elif move == 4:
                await self.attack()
            elif move == 5:
                await self.scout(iteration)

        await self.draw_map(iteration)

        if self.units(UnitTypeId.MARINE).amount == 0 and self.units(UnitTypeId.SCV).amount == 0:
            await self.client.leave()

    async def build_depos(self) -> None:
        """Build a supply depot when low on supply (renamed from build_depos)."""
        if (
            self.supply_left < 5
            and not self.already_pending(UnitTypeId.SUPPLYDEPOT)
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            ):
            try:
                cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
                await self.build(UnitTypeId.SUPPLYDEPOT
                                 , near=cc.position.towards(self.game_info.map_center, 8)
                                 )
            except Exception:
                print("Brak centrum dowodzenia")

            try:
                depot = self.structures(UnitTypeId.SUPPLYDEPOT)[-1]
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
            except Exception as e:
                print(f"Brak magazynu do obnizenia - {e}")

    async def build_barracks(self) -> None:
        """Construct a barracks when possible."""
        try:
            cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
        except Exception:
            print("Nie mozna budowac barakow, zadne cc nie istnieje")
            print("Przerywanie gry -> raczej male szanse na wygrane")
            await self.client.leave()
            return

        try:
            barracks_count = self.structures(UnitTypeId.BARRACKS).amount
            command_center_count = self.structures(UnitTypeId.COMMANDCENTER).amount
            if (
                self.tech_requirement_progress(UnitTypeId.BARRACKS)
                and self.can_afford(UnitTypeId.BARRACKS)
                and (barracks_count / command_center_count) < 3
                ):
                try:
                    await self.build(UnitTypeId.BARRACKS, 
                                     near=cc.position.towards(
                                         self.game_info.map_center, 
                                         random.randint(5, 25))
                                     )
                except Exception:
                    return
        except Exception as e:
            print(f"Nie mozna zbudowac koszar -  {e}")

    async def expand(self) -> None:
        """Expand base logic - build supply depot, train SCVs, or expand command center."""
        try:
            action_taken = False

            # Build a supply depot if supply is low
            if self.supply_left < 4:
                if self.already_pending(UnitTypeId.SUPPLYDEPOT) == 0:
                    if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                        await self.build(UnitTypeId.SUPPLYDEPOT, near=random.choice(self.townhalls))
                        action_taken = True

            # Train SCVs if a supply depot wasn't build
            if not action_taken:
                for cc in self.townhalls:
                    nearby_workers = len(self.workers.closer_than(10, cc))
                    if nearby_workers < 16 and cc.is_idle and self.can_afford(UnitTypeId.SCV):
                        cc.train(UnitTypeId.SCV)
                        action_taken = True

            # Expand to a new command center if nothing else was done
            if not action_taken:
                if self.already_pending(UnitTypeId.COMMANDCENTER) == 0:
                    if self.can_afford(UnitTypeId.COMMANDCENTER):
                        await self.expand_now()

        except Exception as e:
            print(e)


    async def train_marine(self) -> None:
        """Train a marine if possible."""
        try:
            for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
                if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                    barracks.train(UnitTypeId.MARINE)
        except Exception as e:
            print(f"Nie mozna trenowac marine {e}")

    async def train_scv(self) -> None:
        """Train an SCV from any idle command center."""
        try:
            for cc in self.townhalls(UnitTypeId.COMMANDCENTER).ready.idle:
                if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0:
                    cc.train(UnitTypeId.SCV)
        except Exception as e:
            print(f"Nie mozna trenowac SCV - {e}")

    async def attack(self) -> None:
        """Attack with idle marines according to priority rules."""
        try:
            idle_marines = self.units(UnitTypeId.MARINE).idle
            for marine in idle_marines:
                # Attack nearby enemy units first
                nearby_enemies = self.enemy_units.closer_than(12, marine)
                if nearby_enemies:
                    marine.attack(random.choice(nearby_enemies))

                # If no nearby units, attack nearby enemy structures
                nearby_structures = self.enemy_structures.closer_than(12, marine)
                if nearby_structures:
                    marine.attack(random.choice(nearby_structures))

                # If nothing nearby enemy structures, attack any enemy unit
                if self.enemy_units:
                    marine.attack(random.choice(self.enemy_units))

                # If no enemy units, attack any enemy structure
                if self.enemy_structures:
                    marine.attack(self.enemy_structures.random)

                # If nothing exists, scout the map
                x = random.randint(0, self.map_width)
                y = random.randint(0, self.map_height)
                marine.attack(position.Point2(position.Pointlike((x, y))))

        except Exception as e:
            print(f"Blad ataku - {e}")


    async def scout(self, iteration: int) -> None:
        
        """Send an SCV to scout near the enemy start periodically."""
        try: 
            # Scout every n ticks
            if iteration - self.last_scout_iteration > self.ticks_to_scout:
                try:
                    scout_unit = self.units(UnitTypeId.SCV).idle.random
                except Exception:
                    try:
                        scout_unit = self.units(UnitTypeId.SCV).random
                    except Exception as e:
                        print(f"Brak dostepnych SCV - {e}")
                        return

                enemy_location = self.enemy_start_locations[0]
                move_to = self._random_position_variance(enemy_location)

                self.last_scout_iteration = iteration
                scout_unit.move(move_to)

        except Exception as e:
            print(f"Nie mozna wyslac SCV na zwiad - {e}")


    # MAP VISUALS
    async def draw_map(self, iteration: int) -> None:
        """Draw a debug visualization of the game map with units, structures, and resources."""
        global game_map

        game_map = np.zeros((self.game_info.map_size[0], self.game_info.map_size[1], 3), dtype=np.uint8)

        for structure in self.structures:
            if structure.type_id == UnitTypeId.COMMANDCENTER:
                color = [255, 255, 0]  # Yellow
                pos = structure.position
                health_ratio = (structure.health / structure.health_max 
                                if structure.health_max > 0
                                else 0.01
                                )
                
                game_map[int(pos.y), int(pos.x)] = [int(health_ratio * c) for c in color]
            else:
                color = [250, 250, 60]  # Light yellow
                pos = structure.position
                health_ratio = (structure.health / structure.health_max
                                if structure.health_max > 0
                                else 0.01
                                )
                game_map[int(pos.y), int(pos.x)] = [int(health_ratio * c) for c in color]

        # SCV
        for scv in self.units(UnitTypeId.SCV):
            color = [50, 255, 100]
            pos = scv.position
            health_ratio = scv.health / scv.health_max if scv.health_max > 0 else 0.01
            game_map[int(pos.y), int(pos.x)] = [int(health_ratio * c) for c in color]

        # Marine
        for marine in self.units(UnitTypeId.MARINE):
            color = [int(str(marine.tag)[-2:]), 255, 150]
            pos = marine.position
            health_ratio = marine.health / marine.health_max if marine.health_max > 0 else 0.01
            game_map[int(pos.y), int(pos.x)] = [int(health_ratio * c) for c in color]


        # enemy structures
        for enemy_structure in self.enemy_structures:
            color = [0, 255, 255]  # Cyan
            pos = enemy_structure.position
            health_ratio = (enemy_structure.health / enemy_structure.health_max
                            if enemy_structure.health_max > 0
                            else 0.01
                            )
            game_map[int(pos.y), int(pos.x)] = [int(health_ratio * c) for c in color]

        # enemy units
        for enemy_unit in self.enemy_units:
            color = [100, 255, 255]  # Light cyan
            pos = enemy_unit.position
            health_ratio = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.01
            game_map[int(pos.y), int(pos.x)] = [int(health_ratio * c) for c in color]


        # minerals
        for mineral in self.mineral_field:
            pos = mineral.position
            health_ratio = mineral.mineral_contents / 1800
            if mineral.is_visible:
                color = [int(health_ratio * c) for c in [10, 50, 255]]
            else:
                color = [80, 0, 255]
            game_map[int(pos.y), int(pos.x)] = color

        # Display
        resized_map = cv2.resize(game_map, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        flipped_map = cv2.flip(resized_map, 0)
        cv2.imshow("Mapa Gry", flipped_map)
        cv2.waitKey(1)

        # replay if replays enabled
        if self.replays_enabled:
            cv2.imwrite(f"powtorki/{int(time.time())}.png", game_map)

        # Update reward in SARSA
        self._reward_agent(iteration)
        self._save_score()




# Run the game
my_bot = Lemonek()
map = maps.get("AcropolisLE")
game_result = run_game(
    map,
    [Bot(Race.Terran, my_bot), 
        Computer(Race.Terran, Difficulty.Hard)],
        realtime=False)
if str(game_result) == "Result.Victory":
    match_reward = 1000
else:
    match_reward = -1000
print(str(game_result))

# Initialize empty map for SARSA
empty_map = np.zeros((176,184,3), dtype=np.uint8)
game_state_dict = {
    "game_map": empty_map,
    "reward": 0,
    "moves": 1,
    "game_finished": True
    }

# Save initial SARSA
try:
    with open("sarsa.pkl", "wb") as f:
        pickle.dump(game_state_dict, f)
except Exception as exc:
    print(f"Failed to save initial SARSA file: {exc}")

# Close OpenCV windows and exit cleanly
cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()
