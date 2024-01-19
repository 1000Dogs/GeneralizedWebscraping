import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from Search import Search

from pyvis.network import Network




class search_analysis:
    

    #currently testing out saving things as catagories, that may break things so Im not going further with it till other stuff is done

    def __init__(self, pickled_search:Search = None) -> None:
        
        # This is the edge list that stores all of the links in the graph generated from the linked_urls table.
        # It has the columns 'url' and 'link' which are derived directly from linked_urls.
        # And the columns 'domain_url' and 'domain_link' which are the domains paired with the urls.
        # We can grab this directly from pickled search in this case
        # an edge here means that it their was a link to that page and it conained relevent term we cared about
        self.edge_list = pickled_search.get_edge_list(reduced=True)
        self.add_domain_names(self.edge_list,"url")
        self.add_domain_names(self.edge_list,"link")
        
        #all the terms that were considered in this search
        self.analyzed_term_set = pickled_search.relevent_terms

        # This is the term-url paired information. This table is primarily used to lookup which urls are related to which terms.
        # It has the columns 'term', 'url', and 'domain_url' in case you want to do operations by domain.
        self.term_url = pickled_search.get_term_url_pairs_pairs()
        self.add_domain_names(self.term_url, "url")


        self.full_graph : nx.DiGraph = nx.from_pandas_edgelist(self.edge_list, source="url", target="link", create_using=nx.DiGraph)
        self.full_graph.add_nodes_from(self.term_url["url"].to_list())
        

        # The analysis dataframes are just for the purpose of storing computed information for quick reference and comparison.
        # The append methods add to these data frames.
        self.term_anal_df = pd.DataFrame(data = self.analyzed_term_set, columns=["term"])
        self.url_anal_df = self.term_url[["url", "domain_url"]].drop_duplicates()

        self.domain_anal_df = self.url_anal_df[["domain_url"]].drop_duplicates()
        # self.rel_searches = pd.DataFrame(searches.intersection(self.url_anal_df["url"]), columns=["url"])


    def pickle_data(self, folder_location:str):
        pd.to_pickle(self.edge_list, folder_location + '/edge_list.pickle')
        pd.to_pickle(self.term_url, folder_location + '/term_url.pickle')
        pd.to_pickle(self.term_anal_df, folder_location + '/term_anal_df.pickle')
        pd.to_pickle(self.url_anal_df, folder_location + '/url_anal_df.pickle')
        pd.to_pickle(self.domain_anal_df, folder_location + '/domain_anal_df.pickle')
        pd.to_pickle(self.analyzed_term_set, folder_location + '/analyzed_term_set.pickle')
    
    def read_pickle_data(self, folder_location:str):
        self.edge_list = pd.read_pickle(folder_location + '/edge_list.pickle')
        self.term_url = pd.read_pickle(folder_location + '/term_url.pickle')
        self.term_anal_df = pd.read_pickle(folder_location + '/term_anal_df.pickle')
        self.url_anal_df = pd.read_pickle(folder_location + '/url_anal_df.pickle')
        self.domain_anal_df = pd.read_pickle(folder_location + '/domain_anal_df.pickle')
        self.analyzed_term_set = pd.read_pickle(folder_location + '/analyzed_term_set.pickle')

    

    def outgoing_ratio_per_domain(self, domain:str):
        '''
        Returns the number of outgoing links at a given term divded by the total number of pages at that domain
        '''
        outgoing_links = self.edge_list[(self.edge_list["domain_url"] == domain) & (self.edge_list["domain_link"] != domain)]
        resources_at_domain = self.get_url_by_domain(domain, return_list=False)
        return len(outgoing_links) / resources_at_domain if resources_at_domain != 0 else float('NaN')
    
    # probably done need to test
    def get_term_per_domain(self, unique=True) -> pd.DataFrame:
        '''
        Returns the  terms's that are present for each domain in the edge list. Either the list of them or just the number.
        Uses the terms_mentioned term-url table.
            Parameters:
                unique (bool): By default True. Modifies what is returned. If true, counts the number of unique terms within each domain.
                    If false, counts the number of terms mentions, even if a term is mentioned multiple times.
            Returns:
                terms by domain (pd.DataFrame): Data frame of the results based. Columns are "terms" containg the count
                    and "domain_url".
        '''
        #any domain name that had a term related to it, not all seeds possibly
        #unique specifies whether or not we are counting number of unique term's per page or number of term's mentioned
        
        # Grabbing the table of terms and domains, their will be many repeats since each domain may have many pages that each mention terms
        table = self.term_url[["term","domain_url"]]
        # Group by the url, and count the number of unique items in each sub tables 
        grouped_table = table.groupby(by="domain_url")

        return grouped_table.nunique() if unique else grouped_table.count()

    def get_term_per_url(self, unique=True) -> pd.DataFrame:
        '''
        Returns the terms that are present for each url in the edge list. Either the list of them or just the number.
            Parameters:
                unique (bool): By default True. Modifies what is returned. If true, counts the number of unique terms within each domain.
                    If false, counts the number of terms mentions, even if a term is mentioned multiple times.
            Returns:
                terms by domain (pd.DataFrame): Data frame of the results based. Columns are "term" containg the count
                    and "domain_url".
        '''
        #any domain name that had a term related to it, not all seeds
        #unique specifies whether or not we are counting number of unique terms per page or number of term's mentioned
        
        # Grabbing the table of terms and domains, their will be many repeats since each domain may have many pages that each mention terms
        table = self.term_url[["term","url"]]
        # Group by the url, and count the number of unique items in each sub tables 
        grouped_table = table.groupby(by="url")

        return grouped_table.nunique() if unique else grouped_table.count()

    def resource_per_term(self, term:str, return_list:bool = False, unique_domain:bool = False, restrict_to_found_in:bool = False) -> int | pd.Series:
        '''
        Returns the number of resources at a given term in a data frame.

            Parameters:
                term (str): A string representing the term desired.
                return_list (bool): Controls whether a list of the terms or the number is returned.
                unique_domain (bool): Controls whether the found resources are determined by domain or by url.
            
            Returns:
                either an int or a datadrame with the columns "term", and either "domain" or "url" depending on if unique_domain is True.


        '''
        if not unique_domain:
            table = self.term_url.loc[self.term_url["term"] == term, "url"].drop_duplicates() # their should not be duplicates but I am scared by nature
        elif unique_domain:
            table = self.term_url.loc[self.term_url["term"] == term, "domain_url"].drop_duplicates()
        return table if return_list else table.shape[0]
    
    def num_connected_components_by_term(self, term:str, by_domain:bool=False) -> int:
        '''
        Returns the number of connected components at a given term.

            Parameters:
                term (str): A string representing the term desired.
                by_domain (bool): Controls whether the found connected components are determined by domain or by url graphs.
            
            Returns:
                An int representing the number of connected components.
        '''
        restricted_edges, restricted_nodes = self.get_restricted_edge_set(term=term, by_domain=by_domain)
        g : nx.Graph = nx.from_pandas_edgelist(restricted_edges, source=restricted_edges.columns[0], target=restricted_edges.columns[1], create_using=nx.Graph)
        g.add_nodes_from(restricted_nodes)
        return nx.number_connected_components(g)

    def get_reciprocity(self, term:str=None, by_domain:bool=False) -> float:
        '''
        Generates the reciprocity of the graph of a given term. Returns NaN if invalid graph passed

            Parameters:
                term (str): A string representing the term desired.
                by_domain (bool): Controls whether the Graph is determined by domain or by url.
            
            Returns:
                The reciprocity of a given term graph as a float.
        '''
        s = "domain_url" if by_domain else "url"
        t = "domain_link" if by_domain else "link"

        if term is None:
            g = self.build_graph_from_edgelist(self.edge_list, source=s, target=t, remove_self_loop=False)
        else:
            term_restricted, nodes_restricted = self.get_restricted_edge_set(term, by_domain=by_domain)
            g : nx.DiGraph= nx.from_pandas_edgelist(term_restricted, source=s, target=t, create_using=nx.DiGraph)
            g.add_nodes_from(nodes_restricted)
        
        try:
            return nx.reciprocity(g)
        except:
            return float('nan')
    
    def gen_correlation_matrix(self) -> pd.DataFrame:
        '''
        Generates the correlation matrix of the domain graph
            
            Returns:
                An adjacency matrix where an edge between the ith and jth nodes has a weight equal to the number of times that edge occures over all term graphs.
        '''
        domain_edge_list = self.edge_list[["domain_url","domain_link"]].drop_duplicates()
        domain_edge_list["counts"] = np.zeros(shape=(domain_edge_list.shape[0]))

        for term in self.get_term_set():
            edges, _ = self.get_restricted_edge_set(term, by_domain=True)
            sum = domain_edge_list["domain_url"].isin(edges["domain_url"]) & domain_edge_list["domain_link"].isin(edges["domain_link"])
            domain_edge_list.loc[sum, "counts"] += 1


        domain_node_list = self.url_anal_df["domain_url"].drop_duplicates()


        # adjacency = pd.DataFrame(data=np.zeros((num_nodes,num_nodes)), columns=domain_node_list, index=domain_node_list)

        g : nx.DiGraph = nx.from_pandas_edgelist(domain_edge_list, source="domain_url", target="domain_link", edge_attr="counts", create_using=nx.DiGraph)
        g.add_nodes_from(domain_node_list, weight=0)

        return nx.to_pandas_adjacency(g, weight="counts")

    #graphing methods, assumes proper columns are appended
    def gen_term_per_domain(self, unique = True, stacked = False, show_x=False) -> tuple[plt.figure,plt.axes]:
        fig, ax = plt.subplots()
        if not stacked:  
            sorted = self.get_term_per_domain(unique=unique).sort_values(by="term", ascending=True)
            fig.set_size_inches(10,5)
            ax.bar(np.arange(len(sorted)), sorted["term"], .95)
            if unique:
                ax.set_ylabel("#term Reachable")
                ax.set_yscale("log")

                ax.set_xlabel("Domains")
                ax.set_title("Unique terms per Domain")  
            else:
                ax.set_ylabel("#term Reachable")
                ax.set_yscale("log")

                ax.set_xlabel("Domains")
                ax.set_title("terms Mentioned per Domain")
            ax.set_xticks(np.arange(len(sorted)), sorted.index, rotation=90)
        else:
            unique_term_per_domain = self.get_term_per_domain(unique=True)
            term_per_domain = self.get_term_per_domain(unique=False)

            sum = unique_term_per_domain["term"] + term_per_domain["term"]
            sorted_index = sum.argsort()

            unique_term_per_domain = unique_term_per_domain.iloc[sorted_index, :]
            term_per_domain = term_per_domain.iloc[sorted_index, :]



            bottom = np.zeros(unique_term_per_domain.shape[0])
            fig.set_size_inches(10,5)
            #plotting first layer
            ax.bar(x = np.arange(len(unique_term_per_domain)), height = unique_term_per_domain["term"], width = 1, bottom=bottom, label = "Unique term Mentions log 10")

            #plotting second layer
            bottom += unique_term_per_domain["term"]
            ax.bar(np.arange(len(term_per_domain)), term_per_domain["term"], 1, bottom=bottom, label = "term mentions per domain log 10")

            ax.legend()

            ax.set_ylabel("#terms")
            ax.set_yscale("log")


            ax.set_xlabel("Domains")
            ax.set_title("terms per Domain, Unique and Mentioned")
            ax.set_xticks(np.arange(len(term_per_domain)), term_per_domain.index, rotation=90)
        
        if not show_x:
            ax.set_xticks([])
        return (fig, ax)

    
    def gen_reciprocity_hist(self) -> tuple[plt.figure, plt.axes]:
        fig, ax = plt.subplots()
        ax.hist(self.term_anal_df["reciprocity"])
        ax.set_ylabel("# Termss")
        ax.set_xlabel("Reciprocity")
        ax.set_title("Reciprocity By Domain per Term")
        return (fig, ax)
    
    def gen_outgoing_hist(self):
        fig, ax = plt.subplots()
        ax.hist(self.domain_anal_df["outgoing_ratio"], bins = 50)
        ax.set_ylabel("# domains")
        ax.set_xlabel("outgoing_ratio")
        ax.set_title("Outgoing Ratio By Domain")
        ax.set_yscale("log")
        return (fig, ax)

    def gen_graph(self, table:pd.DataFrame, x:str, y:str) -> tuple[plt.figure, plt.axes]:
        '''
        takes in an analysis table and then returns the relevent data in a graph, assuming that data is already appended
        '''
        data = table[[x, y]].dropna()
        fig, ax = plt.subplots()

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.scatter(y = data[y], x = data[x])
        return (fig, ax)

    def get_terms_at_url(self, term:str | list[str], return_list:bool = True) -> np.ndarray | int:
        """
        url : can be a string that represents a url in linked_urls or a list[str] of urls
        restrict to found in : a bool that determines wherther we are talking about what terms are mentioned at the page 
        versus, what search this terms was found in

        returns all the terms that this url talks about uniquely
        """
        dataframe_used = self.term_url

        if type(term) != str:
            values = pd.unique(dataframe_used.loc[dataframe_used["url"].isin(term), "term"])
        else:
            values = pd.unique(dataframe_used.loc[dataframe_used["url"] == term, "term"])
        
        return values if return_list else len(values)

    def get_parents_of_url(self, url:str | list[str], return_list:bool = True) -> np.ndarray | int:
        """
        url : can be a string that represents a url in linked_urls or a list[str] of urls

        returns all the nodes that point to this url
        """
        if type(url) != str:
            return pd.unique(self.edge_list.loc[self.edge_list["url"].isin(url), "url"])
        values = pd.unique(self.edge_list.loc[self.edge_list["url"] == url, "url"])
        return values if return_list else len(values)

    def get_terms_at_domain(self, domain:str, return_list:bool=True) -> np.ndarray | int:
        """
        domain : a string that represents a domain

        returns all the terms that this domain talks about uniquely
        """
        terms = pd.unique(self.term_url.loc[self.term_url["domain_url"] == domain, "term"])
        return terms if return_list else len(terms)



    def append_resource_per_term(self, by_domain=False, reappend:bool = False):
        if reappend or not "resource_per_term" in self.term_anal_df.columns:
            self.term_anal_df["resource_per_term"] = self.term_anal_df["term"].map(lambda x: self.resource_per_term(term=x, unique_domain=by_domain))

    def append_num_term_per_url(self, reappend:bool=False):
        if reappend or not "num_term_per_url" in self.url_anal_df.columns:
            self.url_anal_df["num_term_per_url"] = self.url_anal_df["url"].map(lambda url: self.get_terms_at_url(url, return_list=False))

    def append_num_connected_components(self, reappend:bool=False, by_domain:bool=False):
        if reappend or not "num_connected_components" in self.term_anal_df.columns:
            self.term_anal_df["num_connected_components"] = self.term_anal_df["term"].map(lambda term: self.num_connected_components_by_term(term, by_domain=by_domain))

    def append_outgoing_ratio(self, reappend:bool=False):
        if reappend or not "outgoing_ratio" in self.url_anal_df.columns:
            self.domain_anal_df["outgoing_ratio"] = self.domain_anal_df["domain_url"].map(self.outgoing_ratio_per_domain)

    def append_reciprocity(self, reappend:bool=False, by_domain:bool=False):
        if reappend or not "reciprocity" in self.term_anal_df.columns:
            self.term_anal_df["reciprocity"] = self.term_anal_df["term"].map(lambda x : self.get_reciprocity(x, by_domain))


    @staticmethod
    def append_map(table:pd.DataFrame, func, origin, name:str, reappend:bool=False):
        '''
        general use for applying a map to a origin column of a dataframe and then appending that new column
        '''
        if reappend or not name in table.columns:
            table[name] = table[origin].map(func)


    #probably done, need to manually verify somehow
    def get_term_graph(self) -> nx.DiGraph:
        '''
        attempts to build a networkx digraph of the terms. Does this by taking each source url in the edge list,
        grabbing all of its term refs, and then doing the same for its neighbors, building a graph along the way.
        '''
        g = nx.DiGraph()

        urls = self.edge_list["url"]
        for item in urls:
            item_terms = self.get_terms_at_url(item)
            adjacent_terms = self.get_adjacent_term_set(item)

            for source in item_terms:
                for dest in adjacent_terms:
                    g.add_edge(source, dest)
        return g    

    def get_term_set(self) -> np.ndarray:
        '''
        simply returns the unique set of terms in term_url
        '''
        return self.analyzed_term_set

    def get_url_set(self) -> np.ndarray:
        '''
        returns all urls in the total node list
        '''
        return pd.unique(self.url_anal_df["url"].dropna())
    
    def get_domain_set(self) -> np.ndarray:
        '''
        returns all domains in the total node list
        '''
        return pd.unique(self.url_anal_df["domain_url"].dropna())


    def get_adjacent(self, node:str, edge_list_used=None) -> pd.Series:
        """
        Returns a pd dataframe of adjacent urls based on linked_url
        """
        if edge_list_used is None:
            edge_list_used = self.edge_list
        return edge_list_used.loc[edge_list_used["url"]==node, "link"]
    
    def get_url_by_domain(self, domain:str, return_list:bool=True) -> pd.Series:
        """
        Using a domain as the source it grabs all urls with that domain from url_anal_df

        parameters:
            domain (str): The domain the pull urls from.
            return_list (bool): Whether or not to return the list of urls or the number of urls, by default True
        """
        urls = self.url_anal_df.loc[self.url_anal_df["domain_url"] == domain, "url"]
        return urls if return_list else len(urls)

    def get_adjacent_term_set(self, url:str, edge_list_used=None) -> np.ndarray:
        """
        Returns all terms of all links "adjacent" to this one,
        in the sense that a page that contains this term, links to a page containing the other
        """
        if edge_list_used is None:
            edge_list_used = self.edge_list
        neighbors = self.get_adjacent(url, edge_list_used)
        return self.get_terms_at_url(neighbors)
    
    def add_domain_names(self, df:pd.DataFrame, dom_col:str):
        """
        adds domain names to edge_list and term table
        """
        df["domain_" + dom_col] = df[dom_col]
        df["domain_" + dom_col] = df[dom_col].map(search_analysis.domain_scraper, na_action="ignore")
        df["domain_" + dom_col].astype("category")


    def build_graph_from_edgelist(self, edge_list:pd.DataFrame, source:str, target:str, ignore_root:bool=True, remove_self_loop:bool=True):
        graph:nx.DiGraph = nx.from_pandas_edgelist(edge_list, source=source, target=target, create_using=nx.DiGraph)

        to_remove = []
        for node in graph.nodes():
            if type(node) is float and math.isnan(node):
                to_remove.append(node)
        for nan_obj in to_remove:
            graph.remove_node(nan_obj)
        
        if ignore_root and "root" in graph:
            graph.remove_node("root")
        
        if remove_self_loop:
            self_loops = nx.selfloop_edges(graph)
            graph.remove_edges_from(self_loops)


        return graph

    


    #static methods
    @staticmethod
    def domain_scraper(url:str):
        reform = str.removeprefix(url, "https://")
        reform = str.removeprefix(reform, "http://")
        reform = str.removeprefix(reform, "www.")
        suffix_loc = str.find(reform, '/')
        if not (suffix_loc == -1):
            reform = reform[0:suffix_loc]
        return reform
    
    # stuff below is depreciated for now, may reimplement later
    # @staticmethod
    # def build_graph(data:pd.DataFrame, node_col:str, edge_col:str, save_loc = None, grouping_col:bool=None) -> nx.DiGraph:
        '''
        Builds a graph given a table. Assumes that that table has a format such that one column
        can be accepted as nodes, and another as its that nodes edges.
        '''
        g = nx.from_pandas_edgelist(df=data, source=node_col, target=edge_col, create_using=nx.DiGraph())

        if save_loc is not None:
            nx.write_edgelist(g, save_loc)
        return g
   