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



"""

Bot created with help of guides and sources of burnysc and sentdex.

This entire project is for learning purposes of mine and for engineering thesis
The comments are meant for me to describe what a piece of code does so I can consolidate my knowledge

"""


class GigaBot(BotAI):
    
    #co chce robic co kazdy tick, wywala blad jak sie nazwie inaczej niz on_step lmao
    async def on_step(self, iteration):
        
        #distribute_workers mozna poprawic bo jest podobno nieoptymalna
        await self.distribute_workers()
        await self.build_workers()
        await self.build_depos()
        #await self.build_refinery()
        await self.expand()
        await self.army_buildings()
        await self.build_barrack_units()
        await self.attack()
    
    async def build_workers(self):
        for center in self.townhalls(UnitTypeId.COMMANDCENTER).ready.idle:
            if self.can_afford(UnitTypeId.SCV) and self.units(UnitTypeId.SCV).amount < 70: #mozna pokombinowac z dodatkowymi warunkami dla optymalizacji makro - pomysl: jesli total ilosc scv < il. baz (lub jesli sie da ilosc patchy przy bazach) * 3
                center.train(UnitTypeId.SCV)

    async def build_depos(self):
        if self.townhalls(UnitTypeId.COMMANDCENTER).ready.exists: 
            center = self.townhalls(UnitTypeId.COMMANDCENTER).ready.first
            if not self.already_pending(UnitTypeId.SUPPLYDEPOT) and self.supply_left < 6: #to 6 zalezy, potencjal dla AI by sobie wybier
                if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                    await self.build(UnitTypeId.SUPPLYDEPOT, near=center.position.towards(self.game_info.map_center, 5)) #pozycja moze byc pod AI
                    
    async def build_refinery(self):
        for center in self.townhalls(UnitTypeId.COMMANDCENTER).ready:
            gas_deposits = self.vespene_geyser.closer_than(12.0, center)
            for gas_depo in gas_deposits:
                if not self.can_afford(UnitTypeId.REFINERY):
                    break
                worker = self.select_build_worker(gas_depo.position)
                if worker is None:
                    break
                elif not self.units(UnitTypeId.REFINERY).closer_than(1.0, gas_depo).exists:
                    worker.build(UnitTypeId.REFINERY, gas_depo)
    async def expand(self):
        if self.townhalls(UnitTypeId.COMMANDCENTER).amount < 6 and self.minerals >= 400:
            await self.expand_now()
            
    #na razie skupiamy sie na marines, medievacach i maruderach
    async def army_buildings(self):
        center = self.townhalls(UnitTypeId.COMMANDCENTER).first
        if self.can_afford(UnitTypeId.BARRACKS) and self.structures(UnitTypeId.BARRACKS).amount/self.structures(UnitTypeId.COMMANDCENTER).amount < 2:
            await self.build(UnitTypeId.BARRACKS, near=center.position.towards(self.game_info.map_center, 8))
            
        # if self.can_afford(UnitTypeId.STARPORT) and self.structures(UnitTypeId.STARPORT).amount < 1:
        #     await self.build(UnitTypeId.STARPORT, near=center.position.towards(self.game_info.map_center, 10))
    
    async def build_barrack_units(self):
        #na razie budujemy tylko marines
        for barrack in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE) and self.supply_left > 0:
                barrack.train(UnitTypeId.MARINE)
                
    async def attack(self):
        for marine in self.units(UnitTypeId.MARINE).idle:
            if self.enemy_units:
                #zaatakuj jednostki z najwieksza iloscia ground dps
                if marine.weapon_cooldown <= self.client.game_step / 2:
                    enemies_inrange = self.enemy_units.filter(marine.target_in_range)
                    
                    if enemies_inrange:
                        #priorytet na baneling bo oneshotuja oddzialy
                        filtered_enemy_inrange = enemies_inrange.of_type(UnitTypeId.BANELING)
                        if not filtered_enemy_inrange:
                            filtered_enemy_inrange = max(enemies_inrange, key = lambda enemy: enemy.ground_dps)
                        marine.attack(filtered_enemy_inrange)
                    else:
                        closest_enemy = self.enemy_units.closest_to(marine)
                        marine.attack(closest_enemy)
            

run_game(maps.get("AcropolisLE"), [Bot(Race.Terran, GigaBot()), Computer(Race.Zerg, Difficulty.Medium)], realtime=False)
