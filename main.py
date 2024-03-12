import random
import math
import numpy as np
import sys
import pickle
import time
import sc2
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2 import maps
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.position import Point2, Point3
from sc2.ids.ability_id import AbilityId
from typing import List, Tuple
import random
"""

Bot created with help of guides and sources of burnysc and sentdex.

This entire project is for learning purposes of mine and for engineering thesis
The comments are meant for me to describe what a piece of code does so I can consolidate my knowledge

"""


#12.03.2024
# bot daje sobie rade z zergami na hardzie
# wymagane poprawki kolejnosci budowy
# cc musi odlatywac jak juz nie ma mineralow ani wespanu
# problem nadmiaru scv musi byc szybciej rozwiazywany, czasami w ogole nie dziala 
class GigaBot(BotAI):
    
    def select_target(self):
        #wybierz randomowa znana wroga strukture
        targets = self.enemy_structures
        if targets:
            return targets.random.position
        #jesli brak to bierz
        #randomowa wroga znana jednostke
        targets = self.enemy_units
        if targets:
            return targets.random.position

        #jesli brak to bierz
        #pozycje startowa
        if min((unit.distance_to(self.enemy_start_locations[0]) for unit in self.units)) > 5:
            return self.enemy_start_locations[0]

        #jesli brak to bierz
        #losowe pole mineralow
        return self.mineral_field.random.position
    
    
    #co chce robic co kazdy tick, wywala blad jak sie nazwie inaczej niz on_step lmao
    async def on_step(self, iteration):
        
        #distribute_workers mozna poprawic bo jest podobno nieoptymalna
        await self.distribute_workers()
        await self.build_workers()
        await self.build_depos()
        await self.build_refinery()
        await self.expand()
        await self.army_buildings()
        await self.army_buildings_expansions()
        await self.build_barrack_units()
        await self.attack()
        await self.saturate_refinery()
        await self.build_starport_units()
        
        
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

    def build_techlab(self):
        # buduje techlab lub odlatuje jesli nie ma miejsca
        for sp in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if not sp.has_add_on and self.can_afford(UnitTypeId.BARRACKSTECHLAB):
                addon_points = self.barracks_points_to_build_addon(sp.position)
                if all(
                    self.in_map_bounds(addon_point) and self.in_placement_grid(addon_point)
                    and self.in_pathing_grid(addon_point) for addon_point in addon_points
                ):
                    sp.build(UnitTypeId.BARRACKSTECHLAB)
                else:
                    sp(AbilityId.LIFT)
                    self.barrack_position_change()
                    #self.where_its_flying()
                    
    def build_reactor(self):
        # buduje reaktor lub odlatuje jesli nie ma miejsca
        for sp in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if not sp.has_add_on and self.can_afford(UnitTypeId.BARRACKSREACTOR):
                addon_points = self.barracks_points_to_build_addon(sp.position)
                if all(
                    self.in_map_bounds(addon_point) and self.in_placement_grid(addon_point)
                    and self.in_pathing_grid(addon_point) for addon_point in addon_points
                ):
                    sp.build(UnitTypeId.BARRACKSREACTOR)
                else:
                    sp(AbilityId.LIFT)
                    self.barrack_position_change()
                    #self.where_its_flying()
                    
    def where_its_flying(self):
        # pokazuje gdzie leci ale czy to potrzebne?
        for sp in self.structures(UnitTypeId.STARPORTFLYING).filter(lambda unit: not unit.is_idle):
            if isinstance(sp.order_target, Point2):
                p: Point3 = Point3((*sp.order_target, self.get_terrain_z_height(sp.order_target)))
                self.client.debug_box2_out(p, color=Point3((255, 0, 0)))

        
    
    async def build_workers(self):
        for center in self.townhalls(UnitTypeId.COMMANDCENTER).ready.idle:
            if self.can_afford(UnitTypeId.SCV) and self.units(UnitTypeId.SCV).amount < 80: #mozna pokombinowac z dodatkowymi warunkami dla optymalizacji makro - pomysl: jesli total ilosc scv < il. baz (lub jesli sie da ilosc patchy przy bazach) * 3
                center.train(UnitTypeId.SCV)

    async def build_depos(self):
        if self.townhalls(UnitTypeId.COMMANDCENTER).ready.exists: 
            center = self.townhalls(UnitTypeId.COMMANDCENTER).ready.first
            if self.already_pending(UnitTypeId.SUPPLYDEPOT) < 2 and self.supply_left < 6: #to 6 zalezy, potencjal dla AI by sobie wybieral
                if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                    await self.build(UnitTypeId.SUPPLYDEPOT, near=center.position.towards(self.game_info.map_center, 5)) #pozycja moze byc pod AI
                    
    async def build_refinery(self):
        if self.structures(UnitTypeId.BARRACKS).amount > 0:
            for center in self.townhalls(UnitTypeId.COMMANDCENTER).ready:
                gas_deposits = self.vespene_geyser.closer_than(12.0, center)
                for gas_depo in gas_deposits:
                    if not self.can_afford(UnitTypeId.REFINERY) and self.structures(UnitTypeId).amount > 3:
                        break
                    worker = self.select_build_worker(gas_depo.position)
                    if worker is None:
                        break
                    elif not self.units(UnitTypeId.REFINERY).closer_than(1.0, gas_depo).exists:
                        worker.build(UnitTypeId.REFINERY, gas_depo)
                    
    async def saturate_refinery(self):
        for refinery in self.gas_buildings:
            if refinery.assigned_harvesters < refinery.ideal_harvesters:
                worker = self.workers.closer_than(10, refinery)
                if worker:
                    worker.random.gather(refinery)
                    
    async def expand(self):
        if self.townhalls(UnitTypeId.COMMANDCENTER).amount < 4 and self.minerals >= 400:
            await self.expand_now()
            
    #na razie skupiamy sie na marines, medievacach i maruderach
    async def army_buildings(self):
        center = self.townhalls(UnitTypeId.COMMANDCENTER).first
        if self.can_afford(UnitTypeId.BARRACKS) and self.structures(UnitTypeId.BARRACKS).amount/self.structures(UnitTypeId.COMMANDCENTER).amount < 1:
            await self.build(UnitTypeId.BARRACKS, near=center.position.towards(self.game_info.map_center, random.randint(5,15)))
        if self.tech_requirement_progress(UnitTypeId.FACTORY) == 1:    
            if self.can_afford(UnitTypeId.FACTORY) and self.structures(UnitTypeId.FACTORY).amount < 1 and self.structures(UnitTypeId.COMMANDCENTER).amount > 1:
                await self.build(UnitTypeId.FACTORY, near=center.position.towards(self.game_info.map_center, random.randint(15,20)))    
        if self.tech_requirement_progress(UnitTypeId.STARPORT) == 1:    
            if self.can_afford(UnitTypeId.STARPORT) and self.structures(UnitTypeId.STARPORT).amount < 1 and self.structures(UnitTypeId.COMMANDCENTER).amount > 1:
                await self.build(UnitTypeId.STARPORT, near=center.position.towards(self.game_info.map_center, random.randint(20,25)))
        
    async def army_buildings_expansions(self):
        if self.structures(UnitTypeId.BARRACKS).ready and self.structures(UnitTypeId.BARRACKS).amount > 1:
            option = random.getrandbits(1)
            if option == 0:
                self.build_techlab()
            elif option == 1:
                self.build_reactor()
    
    async def build_barrack_units(self):
        for barrack in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                barrack.train(UnitTypeId.MARINE)
            if self.can_afford(UnitTypeId.MARAUDER) and self.supply_left > 1 and self.units(UnitTypeId.MARINE).amount/(self.units(UnitTypeId.MARAUDER).amount + 1) > 2:
                barrack.train(UnitTypeId.MARAUDER)
    async def build_starport_units(self):
        for starport in self.structures(UnitTypeId.STARPORT).ready.idle:
            if self.can_afford(UnitTypeId.MEDIVAC) and self.supply_left > 1:
                starport.train(UnitTypeId.MEDIVAC)
                
    async def attack(self):
        #jesli mamy full to pusz, ale jesli nie to

        if self.supply_army + self.supply_workers > 190:
            #puszuj ale z priorytetem na mocne jednostki
            #enemy_location = self.enemy_start_locations[0]
            #for unit in self.units(UnitTypeId.MARINE):
            for unit in self.units:
                unit.attack(self.select_target())
                enemies_inrange = self.enemy_units.filter(unit.target_in_range)
                if enemies_inrange:
                    #enemies_inrange = self.enemy_units.filter(unit.target_in_range)
                    filtered_enemy_inrange = enemies_inrange.of_type(UnitTypeId.BANELING)
                    if not filtered_enemy_inrange:
                        filtered_enemy_inrange = max(enemies_inrange, key = lambda enemy: enemy.ground_dps)
                    unit.attack(filtered_enemy_inrange)

        #for unit in self.units(UnitTypeId.MARINE).idle:
        for unit in self.units.idle:
            if unit == UnitTypeId.MEDIVAC:
                unit.move(random.choice(self.units(UnitTypeId.MARINE)))
            if self.enemy_units:
                #zaatakuj jednostki z najwieksza iloscia ground dps
                if unit.weapon_cooldown <= self.client.game_step / 2:
                    enemies_inrange = self.enemy_units.filter(unit.target_in_range)
                    
                    if enemies_inrange:
                        filtered_enemy_inrange = enemies_inrange.of_type(UnitTypeId.BANELING)
                        if not filtered_enemy_inrange:
                            filtered_enemy_inrange = max(enemies_inrange, key = lambda enemy: enemy.ground_dps)
                        unit.attack(filtered_enemy_inrange)
                    else:
                        closest_enemy = self.enemy_units.closest_to(unit)
                        unit.attack(closest_enemy)
            

run_game(maps.get("AcropolisLE"), [Bot(Race.Terran, GigaBot()), Computer(Race.Zerg, Difficulty.Hard)], realtime=False)
