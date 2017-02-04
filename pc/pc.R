####################################
#   Auteurs :                      #
#    -> Marine Ruiz      p1208141  #
#    -> Jules  Sauvinet  p1412086  #
#   Date    :                      #
#    -> 23.01.2017                 #
####################################


install.packages("bnlearn")
source("http://bioconductor.org/biocLite.R")
biocLite(c("graph", "Rgraphviz"))
library("bnlearn")


#-------------------------------------------------------------------------------------------------#
#---------                            Prise en main de R                                  --------#
#-------------------------------------------------------------------------------------------------#


# charge la base de donnees alarm
data("alarm")

# infos sur les colonnes (nos variables)
ncol(alarm)
colnames(alarm)

# infos sur les lignes (nos observations)
nrow(alarm)
rownames(alarm)

# dix premières lignes, 5 premières colonnes
alarm[1:10, 1:5]

# lignes 3, 5, 1, colonnes "ANES", "HIST" et "MINV"
alarm[c(3, 5, 1), c("ANES", "HIST", "MINV")]


#-------------------------------------------------------------------------------------------------#
#---------                        Prise en main de bnlearn                                --------#
#-------------------------------------------------------------------------------------------------#

ci.test(x = "PAP", y = "SHNT", z = as.character(NULL), data = alarm, test = "mi")

res = ci.test(x = "PAP", y = "SHNT", z = "PMB", data = alarm, test = "mi")
res$statistic
res$p.value

table(alarm[, "PAP"])
plot(alarm[, "PAP"])
prop.table(table(alarm[, "PAP"]))

table(alarm[, "SHNT"])
plot(alarm[, "SHNT"])
prop.table(table(alarm[, "SHNT"]))

ct = table(alarm[, c("PAP", "SHNT")])
prop.table(ct)
prop.table(ct, margin = 1)
prop.table(ct, margin = 2)

#Vrai si p > 0.05
#Faux sinon
isCI <- function(x1,y1,z1, seuil) {
  res = ci.test(x=x1, y = y1, z= z1, data = alarm, test ="mi");
  if (res$p.value > seuil){
    return(TRUE);
  }
  else {
    return(FALSE);
  }
}

#Tests d'independance conditionnelle avec p_value = 0.05

#STKV ⟂ HR | ∅; ----> Vrai
isCI("STKV", "HR",as.character(NULL), 0.05) 

#STKV ⟂ HR | CO; ---> Faux
isCI("STKV", "HR","CO", 0.05) 

#HR ⟂ CO | ∅;-     -> Faux
isCI("CO", "HR",as.character(NULL), 0.05) 

#HR ⟂ CO | STKV; ---> Faux
isCI("CO", "HR","STKV", 0.05) 

#CO ⟂ STKV | ∅; ---> Faux
isCI("CO", "STKV",as.character(NULL), 0.05)

#CO ⟂ STKV | HR. ---> Faux
isCI("CO", "STKV","HR", 0.05)

#Quelle structure de reseau Bayesien permet d'encoder le modele d'independance entre les trois variables STKV, HR et CO ?
#Une V-Structure SKTV -> CO <- HR

#Inspectez la relation entre STKV et HR --> HR semble indep de STKV p(HR|STKV) == pour tout stkv
mask = rep(TRUE, nrow(alarm))
p = prop.table(table(alarm[mask, c("STKV", "HR")]), margin = 1)
p
plot(p, main="p(y|x)")

#Inspectez la relation entre STKV et HR sachant CO --> HR semble dep de STKV p(HR|STKV,CO) != pour tout stkv et co
mask = alarm[, "CO"] == "HIGH"
p = prop.table(table(alarm[mask, c("STKV", "HR")]), margin = 1)
plot(p, main="p(y|x,z=HIGH)")

mask = alarm[, "CO"] == "LOW"
p = prop.table(table(alarm[mask, c("STKV", "HR")]), margin = 1)
plot(p, main="p(y|x,z=LOW)")

mask = alarm[, "CO"] == "NORMAL"
p = prop.table(table(alarm[mask, c("STKV", "HR")]), margin = 1)
plot(p, main="p(y|x,z=NORMAL)")



#-------------------------------------------------------------------------------------------------#
#---------                           R�seaux Bay�siens                                    --------#
#-------------------------------------------------------------------------------------------------#

# structure
bn = hc(alarm)
graphviz.plot(bn)

# parametres
bn = bn.fit(bn, data = alarm, method = "bayes")
bn[["CO"]]

#inference de P(STKV="HIGH" | HR ="LOW")
cpquery(bn, event = (STKV == "HIGH"), evidence = (HR == "LOW"))

#inference de P(STKV="HIGH" | HR ="LOW", CO ="LOW")
cpquery(bn, event = (STKV == "HIGH"), evidence = (HR == "LOW" & CO == "LOW"))

#METTRE LE BON REPERTOIRE POUR INCLURE
#setwd("7.PGM")
#getwd()
source("includes.R")

p = exact.dist(bn, event = c("STKV", "HR", "CO"), evidence = TRUE)

sum(p["HIGH", "LOW", ]) / sum(p[, "LOW", ])
sum(p["HIGH", "LOW", "LOW"]) / sum(p[, "LOW", "LOW"])

#long a executer
p = exact.dist(bn, event = c("INT", "APL"), evidence = TRUE)

#A l'aide de la fonction exact.dist(), calculez la distribution conditionnelle de HYP sachant STKV, autrement dit p(y|x). 
p = exact.dist(bn, event = c("HYP", "STKV"), evidence = TRUE) 
p = prop.table(p, margin = 2)



#-------------------------------------------------------------------------------------------------#
#---------                               Do-Calculus                                      --------#
#-------------------------------------------------------------------------------------------------#

#Puis, en supposant que le reseau bayesien est causal, calculez la distribution de 
#probabilite de HYP sachant que la valeur de STKV a ete forcee, autrement dit p(y|do(x)). 
#On rappelle la formule d'ajustement pour "supprimer" une variable confondante: p(y|do(x)) = ∑ p(z)p(y|x,z)
p = exact.dist(bn, event = c("HYP", "STKV", "LVV"), evidence = TRUE)
pdo = prop.table(p, margin = c(2,3))
plvv = margin.table(p, margin = 3)
plvv.expand = array(rep(plvv, each=6), dim = c(2, 3, 3))
margin.table(pdo * plvv.expand, c(1, 2))




#-------------------------------------------------------------------------------------------------#
#---------                               L'algorithme PC                                 ---------#
#-------------------------------------------------------------------------------------------------#

rowmatch <- function(A,B) { 
  # Rows in A that match the rows in B
  f <- function(...) paste(..., sep=":")
  if(!is.matrix(B)) B <- matrix(B, 1, length(B))
  a <- do.call("f", as.data.frame(A))
  b <- do.call("f", as.data.frame(B))
  match(b, a)
}

pc <- function(vars, g) {
  #Commencez par initialiser un squelette (graphe non-dirige) complet. 
  #A partir d'un graphe vide, ajoutez un arc non-dirige entre chaque paire de noeuds:
  vars = colnames(alarm)
  g = empty.graph(vars)
  print(paste("Initialisation d'un graphe de ", length(vars)," noeuds"))
  for (x in vars) {
    for (y in setdiff(vars, x)) {
      g = set.edge(g, from = x, to = y)
    }
  }
  print(paste("Initialisation terminee"))
  #Pour chaque paire (ordonnee) de variables X et Y, et pour chaque variable Z adjacente à X, tester X ⟂ Y | Z. 
  #Si la relation est vraie, alors retirer l'arc correspondant;
  #Pour chaque paire (ordonnee) de variables X et Y, et pour chaque ensemble Z de 2 variables adjacentes à X, tester X ⟂ Y | Z. 
  #Si la relation est vraie, alors retirer l'arc correspondant;
  #proceder ainsi de suite avec des ensemble de taille 3, 4, etc. -> generique :
  xs = c()
  ys = c()
  zs = c()
  for (sizeComb in 0:(length(vars)-2)){
    print(paste("Etape ", sizeComb, ": suppression d'arcs avec test d'independance conditionnelle sur des ensembles de taille ", sizeComb))
    sizeNeig = 0
    for (x in vars) {
      for (y in setdiff(g$nodes[[x]]$nbr,x)) {
        t = setdiff(g$nodes[[x]]$nbr,y);
        Z = c()
        if (length(t) >= sizeComb){
          combis = combn(t, sizeComb)
          for (i in 1:ncol(combis)) {
            combi = combis[,i]
            if (isCI(x,y,combi, 0.01)){
              g <- drop.edge(g, from = x, to = y)
              for (co in combi){
                xs = c(xs,x) 
                ys = c(ys,y) 
                zs = c(zs,co) 
              }
            }
          }
        }
        }
      sizeNeigX = length(g$nodes[[x]]$nbr)
      sizeNeig = max(sizeNeig,sizeNeigX)
    }
    graphviz.plot(g)
    if(sizeComb > sizeNeig) {
      break
    }
  }
  
  print(paste("Orientation des arcs, etape 1"))
  noarcs = unique(cbind(xs,ys,zs))
  vstruct1 = c()
  vstruct2 = c()
  for (x in vars) {
    for(y in g$nodes[[x]]$nbr) {
      if (!x %in% g$nodes[[y]]$children && !x %in% g$nodes[[y]]$parents){
        for(z in intersect(g$nodes[[x]]$nbr,g$nodes[[y]]$nbr)){
          if ( is.na(rowmatch(noarcs,c(x,y,z))) == TRUE ){
            g <- set.arc(g, from = x, to = z)
            g <- set.arc(g, from = y, to = z)
            print(paste("setarc1 de ",x, " a ", z));
            vstruct1 = c(vstruct1,x)
            vstruct2 = c(vstruct2,z)
            print(paste("setarc1 de ",y, " a ", z));
            vstruct1 = c(vstruct1,y)
            vstruct2 = c(vstruct2,z)
          }
        }
      }
    }
  }
  
  graphviz.plot(g)
  
  print(paste("Orientation des arcs, etape 2"))
  
  for (x in vars) {
    for(y in g$nodes[[x]]$nbr) {
      if (!y %in%  g$nodes[[x]]$parents && !y %in%  g$nodes[[x]]$children){
        if (length(g$nodes[[y]]$parents) == 0){
          g <- set.arc(g, from = x, to = y);
          print(paste("setarc2 de ",x, " a ", y ));
        }
        else if (length(g$nodes[[x]]$parents) == 0){
          g <- set.arc(g, from = y, to = x)
          print(paste("setarc2 de ",y, " a ", x ));
        }
      }
    }
  }
  
  graphviz.plot(g)
  
  print(paste("Orientation des arcs, etape 3"))
  for (i in 1:length(directed.arcs(g)[,1])){
    from = directed.arcs(g)[i,1]
    to   = directed.arcs(g)[i,2]
    if (length(intersect(match(from,vstruct1), match(to, vstruct2)))==0){
      if (length(g$nodes[[from]]$parents) == 0){
        g2 = reverse.arc(g,from = to, to = from, check.cycles = FALSE, check.illegal = FALSE, debug = FALSE)
        if (acyclic(g2, directed = TRUE) == TRUE){
          g = g2
          print(paste("reversearc3 de ",from, " a ", to));
        }
      }
    }
  }
    
  graphviz.plot(g)
    
  print(paste("Orientation des arcs, etape 4"))
  for (x in vars) {
    for(y in g$nodes[[x]]$nbr) {
      if (!y %in%  g$nodes[[x]]$parents && !y %in%  g$nodes[[x]]$children){
        if (length(g$nodes[[y]]$parents) == 0){
          g <- set.arc(g, from = x, to = y);
          print(paste("setarc4 de ",x, " a ", y ));
        }
        else if (length(g$nodes[[x]]$parents) == 0){
          g <- set.arc(g, from = y, to = x)
          print(paste("setarc4 de ",y, " a ", x ));
        }
      }
    }
  }
  graphviz.plot(g)
    
  
  print(paste("Execution terminee"))
}

pc(vars,g)


  


