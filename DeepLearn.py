import random
import math
import numpy as np
import sys
import pickle
import time
import sc2
import cv2
import keras
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2 import maps, position
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units
#from sc2.constants import MARINE, COMMANDCENTER, MARAUDER
from sc2.position import Point2, Point3
from sc2.ids.ability_id import AbilityId
from typing import List, Tuple
import random
import asyncio
#np.set_printoptions(threshold=sys.maxsize)

"""
Bot created with help of guides and sources of burnysc and sentdex.

This entire project is for learning purposes of mine and for engineering thesis
The comments are meant for me to describe what a piece of code does so I can consolidate my knowledge

"""
HEADLESS = False

class GigaBot(BotAI):

    def __init__(self, use_model = False):
        self.ITERATIONS_PER_MINUTE = 165
        self.MAX_WORKERS = 75
        self.time_to_action = 0
        self.train_data = []
        self.use_model = use_model

        self.choices = {0: self.build_depos,
                        1: self.build_barracks,
                        2: self.build_refinery,
                        3: self.expand,
                        4: self.build_upgrade,
                        5: self.train_marine,
                        6: self.train_marauder,
                        7: self.train_scv,
                        8: self.attack_enem_start,
                        9: self.attack_enem_struct,
                        10: self.attack_enem_units,
                        11: self.build_factory,
                        12: self.build_starport}
        if self.use_model:
            print("Using model: ON")
            self.model = keras.models.load_model("GIGABOT_epoch_lr_5_0.001_V1.keras")
        else:
            print("Using model: OFF")

    def find_target(self, state):
        if len(self.enemy_units) > 0:
            return random.choice(self.enemy_units)
        elif len(self.enemy_structures) > 0:
            return random.choice(self.enemy_structures)
        else:
            return self.enemy_start_locations[0]
        
    def random_position_variance(self, enemy_start_pos):
        """Returns a random point close to enemy starting position"""
        x = enemy_start_pos[0] + random.randrange(-4, 4)
        y = enemy_start_pos[1] + random.randrange(-4, 4)

        x = 0 if x<0 else x
        y = 0 if y<0 else y
    
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        goto = position.Point2(position.Pointlike((x,y)))
        return goto
    
    def where_its_flying(self):
        # pokazuje gdzie leci ale czy to potrzebne?
        for sp in self.structures(UnitTypeId.STARPORTFLYING).filter(lambda unit: not unit.is_idle):
            if isinstance(sp.order_target, Point2):
                p: Point3 = Point3((*sp.order_target, self.get_terrain_z_height(sp.order_target)))
                self.client.debug_box2_out(p, color=Point3((255, 0, 0)))

    def barracks_points_to_build_addon(self, sp_position: Point2) -> List[Point2]:
        """ Return all points that need to be checked when trying to build an addon. Returns 4 points. """
        addon_offset: Point2 = Point2((2.5, -0.5))
        addon_position: Point2 = sp_position + addon_offset
        addon_points = [
            (addon_position + Point2((x - 0.5, y - 0.5))).rounded for x in range(0, 2) for y in range(0, 2)
        ]
        return addon_points
        
    def barracks_land_positions(self, sp_position: Point2) -> List[Point2]:
            """ Return all points that need to be checked when trying to land at a location where there is enough space to build an addon. Returns 13 points. """
            land_positions = [(sp_position + Point2((x, y))).rounded for x in range(-1, 2) for y in range(-1, 2)]
            return land_positions + self.barracks_points_to_build_addon(sp_position)    
            
    def barrack_position_change(self):
        # znajdz pozycje gdzie mozna wyladowac
        for sp in self.structures(UnitTypeId.BARRACKSFLYING).idle:
            possible_land_positions_offset = sorted(
                (Point2((x, y)) for x in range(-10, 10) for y in range(-10, 10)),
                key=lambda point: point.x**2 + point.y**2,
                )
            offset_point: Point2 = Point2((-0.5, -0.5))
            possible_land_positions = (sp.position.rounded + offset_point + p for p in possible_land_positions_offset)
            for target_land_position in possible_land_positions:
                land_and_addon_points: List[Point2] = self.barracks_land_positions(target_land_position)
                if all(
                    self.in_map_bounds(land_pos) and self.in_placement_grid(land_pos)
                    and self.in_pathing_grid(land_pos) for land_pos in land_and_addon_points
                ):
                    sp(AbilityId.LAND, target_land_position)
                    break

    def build_techlab(self, br):
        # buduje techlab lub odlatuje jesli nie ma miejsca
        
        if not br.has_add_on and self.can_afford(UnitTypeId.BARRACKSTECHLAB):
            addon_points = self.barracks_points_to_build_addon(br.position)
            if all(
                self.in_map_bounds(addon_point) and self.in_placement_grid(addon_point)
                and self.in_pathing_grid(addon_point) for addon_point in addon_points
            ):
                br.build(UnitTypeId.BARRACKSTECHLAB)
            else:
                br(AbilityId.LIFT)
                self.barrack_position_change()
                self.where_its_flying()

    async def on_end(self, game_result):
        print("||----------------------------||")
        print("||----------------------------||")
        print("||                            ||")
        print("||*** on_end method called ***||")
        print("||                            ||")
        print("||----------------------------||")
        print("||----------------------------||")

        with open("gameout-random-vs-veasy.txt","a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))
        
        if game_result == Result.Victory:
            np.save("C:/Users/filip/Desktop/StarCraft2bocik/train_data/{}.npy".format(str(int(time.time()))), np.array(self.train_data, dtype='object'))
        
    async def on_step(self, iteration):
        self.timetime = self.state.game_loop/22.4
        
        if iteration%10 == 0:
            await self.distribute_workers()
        
        await self.scout()
        await self.intel()
        await self.take_action()
    
    async def build_upgrade(self):
        if self.structures(UnitTypeId.BARRACKS).idle.amount:
            br = self.structures(UnitTypeId.BARRACKS).idle.random
            self.build_techlab(br)

    async def build_refinery(self):
        for center in self.townhalls(UnitTypeId.COMMANDCENTER).ready:
            gas_deposits = self.vespene_geyser.closer_than(12.0, center)
            for gas_depo in gas_deposits:
                if not self.can_afford(UnitTypeId.REFINERY) and self.structures(UnitTypeId).amount > 2:
                    break
                worker = self.select_build_worker(gas_depo.position)
                if worker is None:
                    break
                elif not self.units(UnitTypeId.REFINERY).closer_than(1.0, gas_depo).exists:
                    worker.build(UnitTypeId.REFINERY, gas_depo)
                        
    async def scout(self):
        if len(self.units(UnitTypeId.REAPER)) > 0:
            scout = self.units(UnitTypeId.REAPER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_position_variance(enemy_location)
                scout.move(move_to)

        else:
            for barracks in self.structures(UnitTypeId.BARRACKS).ready.idle:
                if self.can_afford(UnitTypeId.REAPER) and self.supply_left > 0:
                    barracks.train(UnitTypeId.REAPER)
        
    async def build_depos(self):

        if self.supply_left < 6 and self.supply_used >= 14 and not self.already_pending(UnitTypeId.SUPPLYDEPOT):
            cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
            if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                await self.build(UnitTypeId.SUPPLYDEPOT, near=cc.position.towards(self.game_info.map_center, 8))
    
    async def train_scv(self):
        
        if self.can_feed(UnitTypeId.SCV) and self.workers.amount < self.MAX_WORKERS:
            try:
                cc = self.townhalls(UnitTypeId.COMMANDCENTER).idle.random
            except:
                return
            else:
                if cc:
                    cc.train(UnitTypeId.SCV)
    
    # TODO - FUNCKJA DO ROZMIESZCZANIA SCV - DZIEDZICZONA JEST NIEOPTYMALNA
    # 
    # async def distribute_scv(self):
    #   if not self.mineral_field or not self.workers or not self.townhalls.ready:
    #       return   

    async def build_starport(self):
        try:
            cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
        except:
            await self.client.leave()
            raise Exception("Nie mozna budowac starportu, zadne cc nie istnieje")
        else:
            if self.tech_requirement_progress(UnitTypeId.STARPORT) and self.can_afford(UnitTypeId.STARPORT) and self.structures(UnitTypeId.STARPORT).amount/self.structures(UnitTypeId.COMMANDCENTER).amount < 1:
                try:
                    await self.build(UnitTypeId.STARPORT, near=cc.position.towards(self.game_info.map_center, random.randint(5,25)))
                except:
                    return None
                
    async def build_factory(self): # buduj baraki/fabryke/starport
        try:
            cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
        except:
            await self.client.leave()
            raise Exception("Nie mozna budowac fabryki, zadne cc nie istnieje")
        else:
            if self.tech_requirement_progress(UnitTypeId.FACTORY) and self.can_afford(UnitTypeId.FACTORY) and self.structures(UnitTypeId.FACTORY).amount/self.structures(UnitTypeId.COMMANDCENTER).amount < 1:
                try:
                    await self.build(UnitTypeId.FACTORY, near=cc.position.towards(self.game_info.map_center, random.randint(5,25)))
                except:
                    return None
                
    async def build_barracks(self):
        try:
            cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
        except:
            await self.client.leave()
            raise Exception("Nie mozna budowac barakow, zadne cc nie istnieje")
        else:
            if self.tech_requirement_progress(UnitTypeId.BARRACKS) and self.can_afford(UnitTypeId.BARRACKS) and self.structures(UnitTypeId.BARRACKS).amount/self.structures(UnitTypeId.COMMANDCENTER).amount < 3:
                try:
                    await self.build(UnitTypeId.BARRACKS, near=cc.position.towards(self.game_info.map_center, random.randint(5,25)))
                except:
                    return None
                
    async def train_marine(self):
        for barrack in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.can_feed(UnitTypeId.MARINE):
                barrack.train(UnitTypeId.MARINE)

    async def train_marauder(self):
        for barrack in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if barrack.has_techlab and self.can_afford(UnitTypeId.MARAUDER) and self.can_feed(UnitTypeId.MARAUDER):
                barrack.train(UnitTypeId.MARAUDER)
    
    async def expand(self):
        if self.townhalls.amount < 3 and self.can_afford(UnitTypeId.COMMANDCENTER):
            await self.expand_now()
        

            
    # async def intel(self):
    #     map_objects = {UnitTypeId.COMMANDCENTER: [15, (0, 255, 0)],
    #                    UnitTypeId.REFINERY: [7, (30, 225, 0)],
    #                    UnitTypeId.BARRACKS: [10, (60, 195, 0)],
    #                    UnitTypeId.FACTORY: [10, (90, 165, 0)],
    #                    UnitTypeId.STARPORT: [10, (120, 135, 0)],
    #                    UnitTypeId.SUPPLYDEPOT: [3, (150, 105, 0)]}
        
    #     map_units = {  UnitTypeId.SCV: [2, (150, 150, 0)],
    #                    UnitTypeId.MARINE:[2, (150, 200, 0)],
    #                    UnitTypeId.MARAUDER:[3, (150, 210, 0)]
    #     }
    #     # width x height
    #     game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
    #     for structure_type in map_objects:
    #         for structure in self.structures(structure_type).ready:
    #             structure_pos = structure.position
    #             cv2.circle(game_data, (int(structure_pos[0]), int(structure_pos[1])), map_objects[structure_type][0], map_objects[structure_type][1], -1)
    #     for unit_type in map_units:
    #         for unit in self.units(unit_type).ready:
    #             pos = unit.position
    #             cv2.circle(game_data, (int(pos[0]), int(pos[1])), map_units[unit_type][0], map_units[unit_type][1], -1)

    #     main_base_names = ["nexus", "commandcenter", "hatchery"]
    #     for enemy_building in self.enemy_structures:
    #         pos = enemy_building.position
    #         if enemy_building.name.lower() not in main_base_names:
    #             cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
    #     for enemy_building in self.enemy_structures:
    #         pos = enemy_building.position
    #         if enemy_building.name.lower() in main_base_names:
    #             cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

    #     for enemy_unit in self.enemy_units:

    #         if not enemy_unit.is_structure:
    #             worker_names = ["probe",
    #                             "scv",
    #                             "drone"]
      
    #             pos = enemy_unit.position
    #             if enemy_unit.name.lower() in worker_names:
    #                 cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
    #             else:
    #                 cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)    

    #     line_max = 50
    #     mineral_ratio = self.minerals / 1500
    #     if mineral_ratio > 1.0:
    #         mineral_ratio = 1.0


    #     vespene_ratio = self.vespene / 1500
    #     if vespene_ratio > 1.0:
    #         vespene_ratio = 1.0

    #     population_ratio = self.supply_left / self.supply_cap
    #     if population_ratio > 1.0:
    #         population_ratio = 1.0

    #     plausible_supply = self.supply_cap / 200.0

    #     military_weight = len(self.units(UnitTypeId.MARINE)) / (self.supply_cap-self.supply_left)
    #     if military_weight > 1.0:
    #         military_weight = 1.0


    #     cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
    #     cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
    #     cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
    #     cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (25, 255, 25), 3)  # gas / 1500
    #     cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (255, 255, 0), 3)  # minerals minerals/1500

    #     self.flipped_game_data = cv2.flip(game_data, 0)
    #     resized = cv2.resize(self.flipped_game_data, dsize=None, fx=2, fy=2)
    #     cv2.imshow('Intel', resized)
    #     cv2.waitKey(1)
    async def intel(self):

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)


        for unit in self.units.ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))


        for unit in self.enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(UnitTypeId.SCV)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            print(str(e))


        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        #print(self.flipped)

        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)


        if not HEADLESS:
            if self.use_model:
                #cv2.imshow(resized)
                cv2.waitKey(1)
            else:
                #cv2.imshow(resized)
                cv2.waitKey(1)

    async def wait(self):
        wait = random.randrange(5, 100)
        self.time_to_action = self.timetime + wait

    async def attack_enem_units(self):
        if len(self.all_enemy_units) > 0:
            target = self.all_enemy_units.closest_to(self.townhalls.random)
            if target:
                for idle_army in self.units.idle:
                    idle_army.attack(target)
    async def attack_enem_struct(self):
        if len(self.enemy_structures) > 0:
            target = self.enemy_structures.random
            if target:
                for idle_army in self.units.idle:
                    idle_army.attack(target)
    async def attack_enem_start(self):
        target = self.enemy_start_locations[0]
        if target:
            for idle_army in self.units.idle:
                idle_army.attack(target)
    async def take_action(self):
        choices = {0: "build depos",
                   1: "build barracks", 
                   2: "build_refinery",
                   3: "expand",
                   4: "build_upgrade",
                   5: "train_marine()",
                   6: "train_marauder()",
                   7: "train_scv()",
                   8: "attack_enem_start()",
                   9: "attack_enem_struct()",
                   10: "attack_enem_units()",
                   11: "build_factory()",
                   12: "build_starport()"}
        
        if self.timetime > self.time_to_action:
            if self.use_model:
                depos_weight = 1
                barracks_weight = 1
                refinery_weight = 1
                expand_weight = 1
                upgrade_weight = 1
                marine_weight = 1
                marauder_weight = 1
                scv_weight = 1

                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 1])])
                weights = [depos_weight, barracks_weight, refinery_weight, expand_weight, upgrade_weight, marine_weight, marauder_weight, scv_weight, 1, 1, 1, 1, 1]
                weighted_prediction = prediction[0]*weights
                choice = np.argmax(weighted_prediction)
                print('Choice:', choices[choice])
            else:
                depos_weight = 5
                barracks_weight = 3
                refinery_weight = 1
                expand_weight = 15
                upgrade_weight = 4
                marine_weight = 20
                marauder_weight = 6
                scv_weight = 8
                choice_weights = depos_weight*[0]+barracks_weight*[1]+refinery_weight*[2]+expand_weight*[3]+upgrade_weight*[4]+marine_weight*[5]+marauder_weight*[6]+scv_weight*[7]+1*[8]+1*[9]+1*[10]+1*[11]+1*[12]
                choice = random.choice(choice_weights)

            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e))

            y = np.zeros(13)
            y[choice] = 1
            self.train_data.append([y, self.flipped])

    
USING_MODEL = False
if USING_MODEL: 
    print("USING MODEL") 
else: print("USING RANDOM CHOICES")
while True:
    run_game(maps.get("AbyssalReefLE"), [Bot(Race.Terran, GigaBot(use_model=USING_MODEL)), Computer(Race.Zerg, Difficulty.Medium)], realtime=False)

