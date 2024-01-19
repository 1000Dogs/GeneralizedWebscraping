import grequests
from gevent import monkey

import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

import numpy as np
import pandas as pd
import validators
import math
from pyvis.network import Network

from bs4 import BeautifulSoup
from bs4 import SoupStrainer

import requests
from collections import deque
import time
import re


import lxml


import sys
from typing import Callable

from langdetect import detect_langs

class Search:
    blocked_url_prefixes = {}

    def __init__(self, search_name:str, to_search:list, relevent_terms:set, term_identifiers:list, restricted_domain:list, searched:set=None, edge_list:dict[str, set[str]]=None, rejected:set=None, term_url_pairs:dict=None, searches_with_error:set=None) -> None:
        self.search_name = search_name
        self.to_search = deque(to_search)
        self.relevent_terms = set([str.lower(item) for item in relevent_terms])
        
        self.term_identifiers = term_identifiers
        self.restricted_domain = restricted_domain

        self.searched = set() if searched is None else searched
        self.edge_list : dict[str, set[str]] = dict() if edge_list is None else edge_list
        self.rejected = set() if rejected is None else rejected # list of things we would never ever want to search again in the future
        self.term_url_pairs : dict[str, set[str]] = dict() if term_url_pairs is None else term_url_pairs
        self.searches_with_error = set() if searches_with_error is None else searches_with_error

        self.iteration = 0
        self.num_processed = 0
        self.total_time = 0

        search_log = open(self.search_name + "_log.csv", mode='w')
        search_log.close()
    

    # @staticmethod
    # def remove_blocked_urls(urls:list, filter_function) -> list:  
    #     # filter_func = lambda url: not (any(str.__contains__(url, blocked) for blocked in blocked_urls))
    #     return list(filter(filter_function, urls))
    
    @staticmethod
    def scrape(term_identifiers:list[str], text:str):
        soup_text = str.lower(text)
        rel_terms_found = set()
        for used_regex in term_identifiers:
            new_terms = set(re.findall(used_regex, soup_text, re.IGNORECASE))
            rel_terms_found.update(new_terms)
        return rel_terms_found

    @staticmethod
    def search(restricted_domain, soup:BeautifulSoup, parent):
        links_elems = soup.find_all('a', href=True)
        links = [i.get('href') for i in links_elems]


        abs_links = Search.gen_absolute_links(links, parent_url=parent)
        real_links = list(filter(lambda url: validators.url(url), abs_links))
        
        if len(restricted_domain) > 0:
            domain_filter = lambda x: Search.get_domain(x) in restricted_domain
            real_links = list(filter(domain_filter, real_links))
        return real_links
    
    @staticmethod
    def gen_absolute_link(url:str, parent:str):
        if re.match(r'mailto:', url) or re.match(r'tel:', url) or re.match(r'/?#', url):
        # this is a mailto or tel link, so return something that will be caught elsewhere
        # also "#" indicates it sis a link within the page, which we ignore
            return None
        elif url.startswith('//'):
            # something like '//example.com/' is a link that indicates "keep the protocol," so we fix using parent_url
            return ('http:' if parent.startswith('http:') else 'https:') + url
        elif Search.get_domain(url) == '':
            # relative links will resolve using the parent URL
            # below line should fix the relative link
            return ('https://' if re.match(r'^https?://', parent) else '') + Search.get_domain(parent) + url
        elif url.startswith('?') and Search.get_domain(parent) == 'vuldb.com':
            return 'https://vuldb.com/' + url
        elif not re.match(r'https?://', url):
            # if we get here, the link has a domain, but does not have a schema needed for a later GET request
            # add "https://" which should fix it (should resolve to http:// if HTTPS is not supported)
            return 'https://' + url
        else:
            # below happens when the link is not relative, has domain, has supplied schema
            return url
    
    @staticmethod
    def get_domain(url: str) -> str:
        # remove the leading 'https://' or 'http://' with or without 'www.'
        # then use the split thing
        return re.sub(r'https?://(www\.)?', '', url).split('/')[0]
    
    @staticmethod
    def gen_absolute_links(urls, parent_url) -> set[str]:
        absolute_links = set()
        for link in urls:
            abs = Search.gen_absolute_link(link, parent=parent_url)
            if abs != None:
                absolute_links.add(abs)
        return absolute_links


    def kill_bad_responces(self, responce_list:list[requests.Response]):
        succesful_responces = []
        for responce in responce_list:
            if responce is None:
                continue
            content_length = responce.headers.get("Content-Length")
            if content_length != None and not content_length.isdigit():
                print("this guys had a weird thing: " + responce.url)

            if content_length != None and content_length.isdigit() and int(content_length) > 10000000:
                responce.close()
                self.rejected.add(responce.url)
                continue
            doc_type = responce.headers.get("Content-Type")
            if doc_type != None and doc_type == "unknown/unknown":
                responce.close()
                self.rejected.add(responce.url)
                continue
            if content_length == None and responce.content.__sizeof__() > 10000000:
                responce.close()
                self.rejected.add(responce.url)
                continue
            succesful_responces.append(responce)
        return succesful_responces
    
    def finalize_url(self, r:requests.Response) -> str:
        # url absolutinating goes on here
        url = None
        if len(r.history) == 0:
            url = r.url
        else:
            # uh oh our url was redirected at some point so we need to change some stuff
            # to reflect that the actual url we are dealing with is this one
            # note the original url used to search is already in searched
            # rejected_set.add(r.history[0].url) idk if this is a neccisary line or if its just going to waste space
            url = r.history[-1].url
            
            if url in self.searched or url in self.rejected or url in self.searches_with_error or url in self.to_search:
                r.close()
                return None # if should not search it continue
            self.searched.add(url) # if we should search it lets continue it now
            
            # fix edge list
            for edge in self.edge_list:
                if r.history[0].url in self.edge_list[edge]:
                    self.edge_list[edge].remove(r.history[0].url)
                    self.edge_list[edge].add(url)
        return url

    @staticmethod
    def is_english(text:str, threshold:float = 0.5) -> bool:
        possible_langs = detect_langs(text)
        english_found = False
        for lang in possible_langs:
            if lang.lang == "en" and lang.prob > threshold:
                english_found = True
        return english_found


    def soupnscrape(self, request_list:list[requests.Response], custom_soupers:dict[str,SoupStrainer]=None) -> list[tuple[str, BeautifulSoup, set]]:
        stock_pot = []
        scrapen_time = 0
        soupen_time = 0
        for r in request_list:
            try:
                if r is None:
                    continue
                
                url = self.finalize_url(r)
                if url == None:
                    continue

                scrape_sucess= False
                soup_sucess= False                   


                
                start = time.time()
                terms_found = set()
                rel_terms_found = set()

                
                # try scrape text
                terms_found = Search.scrape(self.term_identifiers, r.text)
                rel_terms_found = set.intersection(terms_found, self.relevent_terms)
                scrapen_time += time.time() - start
                scrape_sucess = True

                
                # reject any links with 0 relevent terms, just continue if has relevent terms but 0 relevent because we may want to search it later
                if len(terms_found) == 0:
                    self.rejected.add(url)
                    r.close()
                    continue
                if len(rel_terms_found) == 0:
                    r.close()
                    continue         

                
                # soup that link, useing lxml parsing because internet said it was faster
                start = time.time()
                soup = None
                domain = Search.get_domain(url)
                used_souper = custom_soupers.get(domain)

                # choose a custom soup strainer for a domain
                if custom_soupers != None and used_souper != None:
                    soup = BeautifulSoup(r.text, 'lxml', parse_only=used_souper)
                else:
                    soup = BeautifulSoup(r.text, 'lxml', parse_only=SoupStrainer(['a', 'html']))
                    # First check lang with html attribute if possible, 
                    # else checks if langauge is english with greater than 50% prob.
                    lang = soup.find("html").get("lang")
                    if lang != None:
                        if lang != "en":
                            self.rejected.add(url)
                            r.close()
                            continue
                    else: 
                        english = Search.is_english(soup.text)
                        if not english:
                            self.rejected.add(url)
                            r.close()
                            continue
                soupen_time += time.time() - start
                soup_sucess = True
                                # if a url has succefully passed through its gauntlet, we will continue
                stock_pot.append((url, soup, rel_terms_found))

            except Exception as e:
                print("error at: " + url + " " + str(e))
                if not scrape_sucess:
                    scrapen_time += time.time() - start
                elif not soup_sucess:
                    soupen_time += time.time() - start
                self.searches_with_error.add(url)
            finally:
                if r != None:
                    r.close()
        print(f"scrapen time = {str(scrapen_time)[0:5]}, soupen time = {str(soupen_time)[0:5]}")
        return stock_pot
    
    def pickle_search_objs(self, log_name):
        pd.to_pickle(self.to_search, log_name + "_to_search.pickle")
        pd.to_pickle(self.searched, log_name + "_searched.pickle")
        pd.to_pickle(self.edge_list, log_name + "_edge_list.pickle")
        pd.to_pickle(self.rejected, log_name + "_rejected.pickle")
        pd.to_pickle(self.term_url_pairs, log_name + "_term_url_pairs.pickle")
        pd.to_pickle(self.searches_with_error, log_name + "_searches_with_error.pickle")
        pd.to_pickle(self.relevent_terms, log_name + "relevent_terms.pickle")

    def get_edge_list(self, reduced = False) -> pd.DataFrame:
        edges = pd.Series(self.edge_list).explode().reset_index()
        edges.rename(columns={"index" : "url", 0 : "link"}, inplace=True)
        return edges if not reduced else edges[edges["link"].isin(edges["url"])].reset_index(drop=True)
    
    def get_term_url_pairs_pairs(self) -> pd.DataFrame:
        term_url = pd.Series(self.term_url_pairs).explode().reset_index()
        term_url.rename(columns={"index" : "url", 0 : "term"}, inplace=True)
        return term_url

    def iterate_search(self, max_deque_limit : int, max_timeout : float, max_accepted_links, custom_soupers:dict[str,SoupStrainer]=None, filter_function:Callable[[str],bool]=None, verbose=True):
        headers = {
        "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"
        }
        
        #some meta data collection
        self.iteration += 1

        total_timer = time.time()
        num_urls = max_deque_limit if len(self.to_search) > max_deque_limit else len(self.to_search)
        urls = [self.to_search.pop() for i in range(num_urls)]
        self.num_processed += num_urls
        num_rejected = 0

        #we are saying that anything that is in this current batch has been searched so it does not get added again for safety
        for url in urls:
            self.searched.add(url)
        
        #removing any blocked urls that made it into the search from the search set
        if filter_function != None:
            urls = list(filter(filter_function, urls))
        num_rejected += num_urls - len(urls)

        # generate requests
        start = time.time()
        rs = (grequests.get(u, headers=headers, timeout=max_timeout) for u in urls)
        responces : list[requests.Response] = grequests.map(rs, exception_handler=lambda x, y: self.searches_with_error.add(x.url), stream=True)
        request_time =  time.time() - start

        # kill all responces with more than 5mb of data if info exists in content_length of header
        succesful_responces = self.kill_bad_responces(responces)
        


        # soup and scrape all relevent information, only keeping stuff with relevent information
        start = time.time()
        stock_pot = self.soupnscrape(request_list=succesful_responces, custom_soupers=custom_soupers)
        soupnscrape_time =  time.time() - start

        # go through each result in the stock pot, and add its relevent information to the differnt sets
        start = time.time()
        for result in stock_pot: # already rejected where relvent terms mentioned == 0 in soup n scrape
            url, soup, terms_found = result

            self.term_url_pairs[url] = terms_found
            found_urls = Search.search(self.restricted_domain, soup, parent=url)

            if(len(found_urls) > max_accepted_links): # add to rejected set any elements that have over a threshold of links
                print(f"Url had {len(found_urls)} links and was rejected: {url}")
                self.rejected.add(url)
                num_rejected += 1
                continue   
            
            # Add all edges to this list, since we dont know what will be relevent we add all and then pair it down later
            # in post processing.
            for new_url in found_urls:
                if self.edge_list.get(url) != None:
                    self.edge_list[url].add(new_url)
                else:
                    self.edge_list[url] = {new_url}
                    
                if (new_url in self.searched) or (new_url in self.rejected) or (new_url in self.searches_with_error) or (new_url in self.to_search):
                    num_rejected += 1
                    continue
                self.to_search.appendleft(new_url)
                
        search_time = time.time() - start

        
        end = time.time()
        timer = end - total_timer
        self.total_time += end - total_timer

        # search persistancy
        start = time.time()
        pd.to_pickle(self, self.search_name + ".pickle")
        pickle_time = time.time() - start

        if verbose:
            # print some meta data as search is going
            print(f"Queue Size = {len(self.to_search)}, Num Searched = {self.num_processed}, Num Rejected = {num_rejected}, "
                    f"Total Time = {str(self.total_time)[0:7]}, time taken = {str(timer)[0:5]}, "
                    f"time per(aggregate) = {str(self.total_time/self.num_processed)[0:5]}, time per(latest): {str(timer/num_urls)[0:5]}, "
                    f"request: {str(request_time)[0:5]}, soupnscrape: {str(soupnscrape_time)[0:5]}, search: {str(search_time)[0:5]}, "
                    f"pickle_time: {str(pickle_time)[0:5]}, projected_remaining: {str(len(self.to_search)*(self.total_time/self.num_processed))[0:6]}")
            #storing meta data to a log
            search_log = open(self.search_name + "_log.csv", mode='a')
            search_log.write(f"{len(self.to_search)},{self.num_processed},{num_rejected},{str(self.total_time)[0:7]},{str(timer)[0:5]},{str(self.total_time/self.num_processed)[0:5]},{str(timer/num_urls)[0:5]},{str(request_time)[0:5]},{str(soupnscrape_time)[0:5]},{str(search_time)[0:5]}, {str(pickle_time)[0:5]}\n")
            search_log.close()

    def build_links(self, max_deque_limit : int = 500, max_timeout : float = 10, num_its = math.inf, max_accepted_links : int = 1000, custom_soupers:dict[str,SoupStrainer]=None, filter_function:Callable[[str],bool]=None):
        print(f"Queue Size = {len(self.to_search)}, Num Searched = {self.num_processed}, Num Rejected 0, "
            f"Total Time = {self.total_time}, time taken = 0, "
            f"time per(aggregate) = 0, time per(latest): 0, "
            f"request: 0, soupnscrape: 0, search: 0, "
            f"pickle_time: 0, projected_remaining: 0")
        while len(self.to_search) != 0 and self.iteration < num_its:
            self.iterate_search(max_deque_limit, max_timeout, max_accepted_links, custom_soupers=custom_soupers, filter_function=filter_function)
    
def main() -> int:

    a = ["https://en.wikipedia.org/wiki/Metric_space"]
    

    # a custom soup strainer is given for the wikipedia domain so that searching does not get stuck in search bars
    custom_soupers = {"en.wikipedia.org":SoupStrainer(attrs={"class":"vector-body"}, multi_valued_attributes=None)}
    blocked_urls = ["https://en.wikipedia.org/w/index.php?"]
    filter_function = lambda url: not (any(str.__contains__(url, blocked) for blocked in blocked_urls))
    
    # search_obj : Search = pd.read_pickle("very_small_test.pickle")
    search_obj : Search = Search(
        search_name="very_small_test", 
        to_search=a, 
        relevent_terms={'metric space'}, 
        term_identifiers=[r"metric space"], #can be any regex statement, currently the same as relevent terms field to simplify the example
        restricted_domain=["en.wikipedia.org"])

    
    search_obj.build_links(custom_soupers=custom_soupers, filter_function=filter_function)
    #after each iteration the search is saved, so if their is some crash due to internet errors, the search can be restarted by unpickling the object and calling build links again
    return 1

if __name__ == '__main__':
    main()
    sys.exit()

