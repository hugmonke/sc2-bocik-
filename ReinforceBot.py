import random
import math
import numpy as np
import time
import cv2
import keras
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2 import maps, position
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2, Point3
from sc2.ids.ability_id import AbilityId
from typing import List, Tuple
import random
import tensorflow.python.keras.backend as backend
import tensorflow as tf
import sys
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


#Meta parametry
POWTORKI = True
class LastResort(BotAI):

    # def __init__(self):
    #     self.mineraly = []
    #     self.czas = []
    #     self.staremineraly = 0
    #     self.nowemineraly = 0
    #     self.sumamineralow = 0
    def nagrodz_bota(self, iteracja):  
        nagroda = 0
        try:
            for marine in self.units(UnitTypeId.MARINE):
                if marine.is_attacking and marine.target_in_range:
                    if self.enemy_units.closer_than(10, marine) or self.enemy_structures.closer_than(10, marine):
                        nagroda += 0.02

        except Exception as problem:
            nagroda = 0
            print(problem)

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

        if iteracja % 100 == 0:
            print(f"Iteracja: {iteracja} | Aktualna nagroda: {nagroda} | Ilosc marine: {self.units(UnitTypeId.MARINE).amount}")

        return nagroda
        
    def zapisz_wynik(self, nagroda):
        dane_rozgrywki = {"mapa_gry": mapa_gry, "nagroda": nagroda, "ruchy": None, "rozgrywka_skonczona": False}
        with open('sarsa.pkl', 'wb') as f:
            pickle.dump(dane_rozgrywki, f)

    def random_position_variance(self, enemy_start_pos):
        """Returns a random point close to enemy starting position"""

        kierunek = random.randint(0, 4)
        if kierunek == 0:
            #print('lewo')
            x = enemy_start_pos[0] - 40
            y = enemy_start_pos[1] - 15
        elif kierunek == 4:
            #print('prawo')
            x = enemy_start_pos[0] + 30
            y = enemy_start_pos[1] + 10
        else:
            #print('srodek')
            x = enemy_start_pos[0]
            y = enemy_start_pos[1]

        x = 0 if x<0 else x
        y = 0 if y<0 else y
    
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        goto = position.Point2(position.Pointlike((x,y)))
        return goto
    # Do przyszlego uzycia?
    # def assign_medivac(self):
    #     for med in self.units(UnitTypeId.MEDIVAC).idle:
    #         med.move(random.choice(self.units(UnitTypeId.MARINE)))
    async def on_end(self, game_result):
        print("||----------------------------||")
        print("||----------------------------||")
        print("||                            ||")
        print("||*** on_end method called ***||")
        print("||                            ||")
        print("||----------------------------||")
        print("||----------------------------||")
        print(self.minerals)
        # np.savetxt("mineralylosowe.txt", self.mineraly, newline=" ")
        # np.savetxt("czas.txt", self.czas, newline=" ")
        with open("czasiwynik.txt","a") as f:
            f.write("Model {} - time {} \n".format(game_result, self.state.game_loop/22.4))

    async def on_step(self, iteracja):
        # self.staremineraly = self.nowemineraly
        # self.nowemineraly = self.minerals
        # if self.nowemineraly > self.staremineraly:
        #     roznica = self.nowemineraly-self.staremineraly
        #     self.sumamineralow += roznica
        #     self.mineraly.append(self.sumamineralow)
        #     self.czas.append(self.state.game_loop/22.4)
        sa_ruchy = False
        while not sa_ruchy:
            try:
                with open('sarsa.pkl', 'rb') as f: #SARSA - State–action–reward–state–action: algorithm for learning a Markov decision process policy
                    
                    sarsa = pickle.load(f)

                    if sarsa['ruchy'] is None:
                        sa_ruchy = False
                    else:
                        sa_ruchy = True
            except:
                pass       
            
        if iteracja%50 == 0:
            await self.distribute_workers()

        ruch = sarsa['ruchy']
        #ruch = np.random.choice([0,1,2,3,4,5])
        if iteracja%5 == 0:
            if ruch == 0:
                await self.expand()
            if ruch == 1:
                await self.build_barracks()
            if ruch == 2:
                await self.train_marine()
            if ruch == 3:
                await self.train_scv()
            if ruch == 4:
                await self.atakuj()
            if ruch == 5:
                await self.scout(iteracja)

             
        await self.rysujmape(iteracja) 
        if self.units(UnitTypeId.MARINE).amount == 0 and self.units(UnitTypeId.SCV).amount == 0:
            await self.client.leave()

    async def build_depos(self):
        if self.supply_left < 5 and not self.already_pending(UnitTypeId.SUPPLYDEPOT) and self.can_afford(UnitTypeId.SUPPLYDEPOT):
            try:
                cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
                await self.build(UnitTypeId.SUPPLYDEPOT, near=cc.position.towards(self.game_info.map_center, 8))
            except:
                print("Brak centrum dowodzenia")

            
            try:
                depo = self.structures(UnitTypeId.SUPPLYDEPOT)[-1]
                depo(AbilityId.MORPH_SUPPLYDEPOT_LOWER)
            except:
                print("Brak magazynu")
  
    async def build_barracks(self):
        try:
            cc = self.townhalls(UnitTypeId.COMMANDCENTER).random
        except:
            await self.client.leave()
            print("Nie mozna budowac barakow, zadne cc nie istnieje")
        else:
            if self.tech_requirement_progress(UnitTypeId.BARRACKS) and self.can_afford(UnitTypeId.BARRACKS) and self.structures(UnitTypeId.BARRACKS).amount/self.structures(UnitTypeId.COMMANDCENTER).amount < 3:
                try:
                    await self.build(UnitTypeId.BARRACKS, near=cc.position.towards(self.game_info.map_center, random.randint(5,25)))
                except:
                    return
    async def expand(self):
        try:
            found_something = False
            if self.supply_left < 4:
                if self.already_pending(UnitTypeId.SUPPLYDEPOT) == 0:
                    if self.can_afford(UnitTypeId.SUPPLYDEPOT):
                        await self.build(UnitTypeId.SUPPLYDEPOT, near=random.choice(self.townhalls))
                        found_something = True

            if not found_something:

                for cc in self.townhalls:
                    worker_count = len(self.workers.closer_than(10, cc))
                    if worker_count < 16:
                        if cc.is_idle and self.can_afford(UnitTypeId.PROBE):
                            cc.train(UnitTypeId.SCV)
                            found_something = True

            if not found_something:
                if self.already_pending(UnitTypeId.COMMANDCENTER) == 0 and self.can_afford(UnitTypeId.COMMANDCENTER):
                    await self.expand_now()

        except Exception as e:
            print(e)

    async def train_marine(self):
        try:
            self.structures(UnitTypeId.BARRACKS).ready
        except:
            print("Brak koszar")
            return
        for barrack in self.structures(UnitTypeId.BARRACKS).ready:
            if self.can_afford(UnitTypeId.MARINE) and self.can_feed(UnitTypeId.MARINE):
                barrack.train(UnitTypeId.MARINE)

    async def train_scv(self):
        
        if self.can_feed(UnitTypeId.SCV) and self.workers.amount < 49:
            try:
                cc = self.townhalls(UnitTypeId.COMMANDCENTER).idle.random
            except:
                return
            else:
                if cc:
                    cc.train(UnitTypeId.SCV) 

    async def atakuj(self):
        print("atak wywolany")
        try:
            for marine in self.units(UnitTypeId.MARINE).idle:
                # Najpierw atakuj wrogich zolnierzy w twoim otoczeniu
                if self.enemy_units.closer_than(12, marine):
                    marine.attack(random.choice(self.enemy_units.closer_than(12, marine)))
                # Jesli nie ma zolnierzy, to atakuj wrogie budynki w twoim otoczeniu
                elif self.enemy_structures.closer_than(12, marine):
                    marine.attack(random.choice(self.enemy_structures.closer_than(12, marine)))
                # Jesli nic nie ma w twoim otoczeniu, to atakuj losowego zolnierza
                elif self.enemy_units:
                    marine.attack(random.choice(self.enemy_units))
                # Jesli nie ma wrogich jednostek, to atakuj losowe wrogie budynki
                elif self.enemy_structures:
                    print("if na struktury")
                    marine.attack(self.enemy_structures.random)
                else:
                        print("else na losowo")
                        x = random.randint(0, 128)
                        y = random.randint(0, 128)

                        x = 0 if x<0 else x
                        y = 0 if y<0 else y
                    
                        if x > self.game_info.map_size[0]:
                            x = self.game_info.map_size[0]
                        if y > self.game_info.map_size[1]:
                            y = self.game_info.map_size[1]
                        marine.attack(position.Point2(position.Pointlike((x,y))))
                        czas_szukania += 1
        except Exception as problem:
            print(problem)      
    
    async def scout(self, iteracja):
        try: 
            self.ostatni
        except:
            self.ostatni = 0
        
        if iteracja - self.ostatni > 180:
            try:
                scout = self.units(UnitTypeId.SCV).idle.random
            except:
                try:
                    scout = self.units(UnitTypeId.SCV).random
                except:
                    print("Brak SCV")
                    return
            enemy_location = self.enemy_start_locations[0]
            move_to = self.random_position_variance(enemy_location)
            self.ostatni = iteracja
            scout.move(move_to)

    async def rysujmape(self, iteracja):     
        
        # Inicjalizacja mapy giereczki
        global mapa_gry
        mapa_gry = np.zeros((self.game_info.map_size[0], self.game_info.map_size[1], 3), np.uint8)

        # Rysowanie moich budynkow na mapie
        for wrogi_budynek in self.structures:
            # Jesli centrum dowodzenia
            if wrogi_budynek.type_id == UnitTypeId.COMMANDCENTER:
                kolory_cc = [255, 255, 0]
                pozycja_cc = wrogi_budynek.position
                if wrogi_budynek.health_max > 0:
                    procent_zycia_cc = wrogi_budynek.health / wrogi_budynek.health_max 
                else:
                    procent_zycia_cc = 0.01
                mapa_gry[int(pozycja_cc.y), int(pozycja_cc.x)] = [int(procent_zycia_cc * rgb) for rgb in kolory_cc]
            # Jesli moj budynek inny niz centrum dowodzenia
            else:
                kolory_budynku = [250, 250, 60]
                pozycja_budynku = wrogi_budynek.position
                if wrogi_budynek.health_max > 0:
                    procent_zycia_budynku = wrogi_budynek.health / wrogi_budynek.health_max 
                else:
                    procent_zycia_budynku = 0.01
                mapa_gry[int(pozycja_budynku.y), int(pozycja_budynku.x)] = [int(procent_zycia_budynku * rgb) for rgb in kolory_budynku]

        # Rysowanie moich jednostek na mapie
        # SCV
        for moja_jednostka in self.units(UnitTypeId.SCV):
            kolory_scv = [50, 255, 100]
            pozycja_scv = moja_jednostka.position
            if moja_jednostka.health_max > 0:
                procent_zycia_jednostki = moja_jednostka.health / moja_jednostka.health_max 
            else:
                procent_zycia_jednostki = 0.01
            mapa_gry[int(pozycja_scv.y), int(pozycja_scv.x)] = [int(procent_zycia_jednostki * rgb) for rgb in kolory_scv]

        # Marine 
        for i in range(0, len(self.units(UnitTypeId.MARINE))):
            kolory_marines = [int(str(self.units(UnitTypeId.MARINE)[i].tag)[-2:]), 255, 150]
            pozycja_marine = self.units(UnitTypeId.MARINE)[i].position
            if self.units(UnitTypeId.MARINE)[i].health_max > 0:
                procent_zycia_marine = self.units(UnitTypeId.MARINE)[i].health / self.units(UnitTypeId.MARINE)[i].health_max 
            else:
                procent_zycia_marine = 0.01
            mapa_gry[int(pozycja_marine.y), int(pozycja_marine.x)] = [int(procent_zycia_marine * rgb) for rgb in kolory_marines]
            

        # Rysowanie wrogich budynkow na mapie
        for wrogi_budynek in self.enemy_structures:
            if wrogi_budynek.health_max > 0:
                procent_zycia_bud_wroga = wrogi_budynek.health / wrogi_budynek.health_max 
            else:
                procent_zycia_bud_wroga = 0.01

            kolory_budynku_wroga = [0, 255, 255]
            pozycja_budynku_wroga = wrogi_budynek.position
            mapa_gry[int(pozycja_budynku_wroga.y), int(pozycja_budynku_wroga.x)] = [int(procent_zycia_bud_wroga * rgb) for rgb in kolory_budynku_wroga]

        # Rysowanie wrogich jednostek na mapie
        for wroga_jednostka in self.enemy_units:
            if wroga_jednostka.health_max > 0:
                procent_zycia_jedn_wroga = wroga_jednostka.health / wroga_jednostka.health_max 
            else:
                procent_zycia_jedn_wroga = 0.01
            
            kolory_jedn_wroga = [100, 255, 255]
            pozycja_jedn_wroga = wroga_jednostka.position
            mapa_gry[int(pozycja_jedn_wroga.y), int(pozycja_jedn_wroga.x)] = [int(procent_zycia_jedn_wroga * rgb) for rgb in kolory_jedn_wroga]

        # Rysowanie krysztalu
        for krysztal in self.mineral_field:
            procent_wydobycia_krysztalu = krysztal.mineral_contents / 1800
            pozycja_krysztalu = krysztal.position
            if krysztal.is_visible:
                mapa_gry[int(pozycja_krysztalu.y), int(pozycja_krysztalu.x)] = [int(procent_wydobycia_krysztalu * rgb) for rgb in [10, 50, 255]]
            else:
                mapa_gry[int(pozycja_krysztalu.y), int(pozycja_krysztalu.x)] = [80, 0, 255]

        resized = cv2.resize(mapa_gry, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        flipped = cv2.flip(resized, 0)
        cv2.imshow('Mapa Gry', flipped)
        cv2.waitKey(1)

        if POWTORKI:
            cv2.imwrite(f"powtorki/{int(time.time())}.png", mapa_gry)

        self.zapisz_wynik(nagroda=self.nagrodz_bota(iteracja))


#def main():
wynik_rozgrywki = run_game(maps.get("TEST1"), [Bot(Race.Terran, LastResort()), Computer(Race.Terran, Difficulty.Hard)], realtime=False)
print('main reinforce bot')

if str(wynik_rozgrywki) == "Result.Victory":
    nagroda_za_mecz = 1000
else:
    nagroda_za_mecz = -1000
print(str(wynik_rozgrywki))


# with open("procesuczenia.txt","a") as f:
#     f.write(f"{wynik_rozgrywki}\n")

pusta_mapa = np.zeros((128, 128, 3), dtype=np.uint8)
#obserwacje = pusta_mapa
dane_rozgrywki = {"mapa_gry": pusta_mapa, "nagroda": nagroda_za_mecz, "ruchy": None, "rozgrywka_skonczona": True}
with open('sarsa.pkl', 'wb') as f:
    pickle.dump(dane_rozgrywki, f)

cv2.destroyAllWindows()
cv2.waitKey(1)
time.sleep(3)
sys.exit()

#main()
