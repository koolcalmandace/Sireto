"""
naf_taxonomy.py — Official INSEE NAF Rév. 2 Nomenclature Descriptions.

Authoritative reference mapping French NAF/APE sub-classes to their official
INSEE economic descriptions. Used by the Semantic Vector Engine to project
commercial and artisanal CRM queries directly into continuous activity space.
"""

from __future__ import annotations
from typing import Dict

# Dictionary of core INSEE NAF Rév. 2 sub-classes and their official French economic descriptions
# Spans all primary sectors where individual businesses, artisans, liberal professions,
# and commercial retail operate.
INSEE_NAF_DESCRIPTIONS: Dict[str, str] = {
    # -----------------------------------------------------------------------
    # Section C : Industrie manufacturière & Artisanat de fabrication
    # -----------------------------------------------------------------------
    "10.71A": "Fabrication industrielle de pain et de patisserie fraiche",
    "10.71B": "Cuisson de produits de boulangerie",
    "10.71C": "Boulangerie et boulangerie patisserie artisanale fabrication vente de pain viennoiserie",
    "10.71D": "Patisserie artisanale chocolaterie confiserie fabrication de gateaux",
    "10.72Z": "Fabrication de biscuits biscottes et patisseries de conservation",
    "10.82Z": "Fabrication de cacao chocolat et produits de confiserie",
    "10.11Z": "Transformation et conservation de la viande boucherie abattage",
    "10.13A": "Fabrication industrielle de produits a base de viande charcuterie",
    "10.13B": "Charcuterie artisanale traiteur fabrication vente de produits charcutiers",
    "14.13Z": "Fabrication de vetements de dessus pret a porter confection habillement",
    "14.14Z": "Fabrication de vetements de dessous lingerie sous vetements bonneterie",
    "15.20Z": "Fabrication de chaussures bottes maroquinerie cordonnerie",
    "18.12Z": "Autre imprimerie labeur reprographie serigraphie impression",
    "20.42Z": "Fabrication de parfums et de produits pour la toilette cosmetique cosmetologie",
    "31.09B": "Fabrication d autres meubles et menuiserie d agencement ebenisterie",
    "32.50A": "Fabrication de materiel medico chirurgical et dentaire prothesiste dentaire prothese",
    "33.12Z": "Reparation de machines et equipements mecaniques maintenance industrielle",
    "33.14Z": "Reparation d equipements electriques transformateurs moteurs generateurs",

    # -----------------------------------------------------------------------
    # Section F : Bâtiment & Travaux Publics (Artisans du bâtiment)
    # -----------------------------------------------------------------------
    "41.20A": "Construction de maisons individuelles constructeur batiment pavillon",
    "41.20B": "Construction d autres batiments gros oeuvre maconnerie generale",
    "43.21A": "Travaux d installation electrique dans tous locaux electricien electricite batiment domotique courant fort faible",
    "43.22A": "Travaux d installation d eau et de gaz en tous locaux plomberie plombier sanitaire depannage fuite tuyauterie",
    "43.22B": "Travaux d installation d equipements thermiques et de climatisation chauffagiste chauffage pompe a chaleur climatiseur",
    "43.29A": "Travaux d isolation thermique acoustique phonique calorifugeage",
    "43.29B": "Autres travaux d installation serrurerie metallerie fermetures volets stores portes automatiques",
    "43.31Z": "Travaux de platrerie platre plaquiste faux plafonds cloisons",
    "43.32A": "Travaux de menuiserie bois et pvc fenetres portes parquets placards agencement",
    "43.32B": "Travaux de menuiserie metallique et serrurerie aluminium portails verandas",
    "43.33Z": "Travaux de revetement des sols et des murs carrelage carreleur faience sols souples",
    "43.34Z": "Travaux de peinture et vitrerie peintre en batiment ravalement facade tapisserie papier peint",
    "43.39Z": "Autres travaux de finition renovation interieure bricolage multiservices",
    "43.91A": "Travaux de charpente charpentier ossature bois",
    "43.91B": "Travaux de couverture par elements couvreur toiture zinguerie etancheite velux",
    "43.99A": "Travaux d etancheification etancheite terrasse fondation cuvelage",
    "43.99C": "Travaux de maconnerie generale et gros oeuvre de batiment macon enduit dallage",

    # -----------------------------------------------------------------------
    # Section G : Commerce, Réparation automobile et motocycles
    # -----------------------------------------------------------------------
    "45.11Z": "Commerce de voitures et de vehicules automobiles legers concessionnaire vente autos",
    "45.20A": "Entretien et reparation de vehicules automobiles legers garage mecanique carrosserie pneu pneus pneumatiques controle technique",
    "45.20B": "Entretien et reparation d autres vehicules automobiles poids lourds camions",
    "45.32Z": "Commerce de detail d equipements automobiles pieces detachees et accessoires auto",
    "45.40Z": "Commerce et reparation de motocycles motos scooters quads cyclomoteurs",
    "47.11A": "Commerce de detail de produits surgeles congelateur",
    "47.11B": "Commerce d alimentation generale epicerie superepice mini marche",
    "47.11C": "Superettes magasin libre service alimentaire",
    "47.11D": "Supermarches grande distribution alimentaire",
    "47.11F": "Hypermarches",
    "47.21Z": "Commerce de detail de fruits et legumes primeur",
    "47.22Z": "Commerce de detail de viandes et de produits a base de viande boucherie charcuterie triperie",
    "47.23Z": "Commerce de detail de poissons crustaces et mollusques poissonnerie ecailler",
    "47.24Z": "Commerce de detail de pain patisserie et confiserie boulanger patissier depot de pain",
    "47.25Z": "Commerce de detail de boissons caviste vin biere spiritueux",
    "47.26Z": "Commerce de detail de produits a base de tabac bureau de tabac civette",
    "47.30Z": "Commerce de detail de carburants en magasin specialise station service essence lavage auto",
    "47.51Z": "Commerce de detail de textiles mercerie tissus rideaux",
    "47.52A": "Commerce de detail de quincaillerie peintures et verres bricolage materiaux",
    "47.54Z": "Commerce de detail d appareils electromenagers petit gros electromenager",
    "47.59A": "Commerce de detail de meubles magasin de mobilier literie ameublement",
    "47.71Z": "Commerce de detail d habillement en magasin specialise vetements pret a porter mode femme homme boutique lingerie",
    "47.72A": "Commerce de detail de la chaussure magasin de souliers bottes baskets",
    "47.72B": "Commerce de detail de maroquinerie et d articles de voyage sacs valises",
    "47.73Z": "Commerce de detail de produits pharmaceutiques en magasin specialise pharmacie officine parapharmacie medicaments ordonnance",
    "47.74Z": "Commerce de detail d articles medicaux et orthopediques materiel medical bandagiste protheses",
    "47.75Z": "Commerce de detail de parfumerie et de produits de beaute cosmetique parfumerie",
    "47.76Z": "Commerce de detail de fleurs plantes graines engrais animaux de compagnie fleuriste jardinerie horticulture graineterie animalerie",
    "47.77Z": "Commerce de detail d articles d horlogerie et de bijouterie bijoutier joaillier montre horloge",
    "47.78A": "Commerces de detail d optique opticien lunettes lentilles de contact montures verres",
    "47.78C": "Autres commerces de detail specialises photo armes souvenirs coutellerie",
    "47.79Z": "Commerce de detail de biens d occasion brocante antiquite friperie depot vente",
    "47.91A": "Vente a distance sur catalogue general e commerce vente en ligne internet",

    # -----------------------------------------------------------------------
    # Section H : Transports & Logistique
    # -----------------------------------------------------------------------
    "49.32Z": "Transports de voyageurs par taxis taxi artisan vtc vehicule de transport avec chauffeur",
    "49.39A": "Transports routiers reguliers de voyageurs autocars lignes d autobus",
    "49.39B": "Autres transports routiers de voyageurs excursions touristiques transport scolaire",
    "49.41A": "Transports routiers de fret interurbains camionnage transport de marchandises",
    "49.41B": "Transports routiers de fret de proximite livraison express camionnette",
    "49.42Z": "Services de demenagement demenageur garde meubles",
    "52.21Z": "Services auxiliaires des transports terrestres parking stationnement peage remorquage depannage",
    "53.20Z": "Autres activites de poste et de courrier coursier livraison colis express",

    # -----------------------------------------------------------------------
    # Section I : Hébergement & Restauration
    # -----------------------------------------------------------------------
    "55.10Z": "Hotels et hebergement similaire hotelier chambre d hotel pension",
    "55.20Z": "Hebergement touristique et autre hebergement de courte duree gite meuble de tourisme chambre d hotes residence de vacances",
    "55.30Z": "Terrains de camping et parcs pour caravanes ou vehicules de loisirs hotellerie de plein air camping caravaning mobil home parc residentiel",
    "56.10A": "Restauration traditionnelle restaurant brasserie auberge bistrot table",
    "56.10C": "Restauration de type rapide fast food snack pizzeria sandwicherie kebab burger food truck",
    "56.21Z": "Services des traiteurs traiteur receptions banquets buffets",
    "56.30Z": "Debits de boissons cafe bar pub brasserie salon de the discotheque",

    # -----------------------------------------------------------------------
    # Section J : Information, Informatique & Communication
    # -----------------------------------------------------------------------
    "59.11B": "Production de films institutionnels et publicitaires audiovisuel video",
    "62.01Z": "Programmation informatique developpeur logiciel web application dev",
    "62.02A": "Conseil en systemes et logiciels informatiques consultant it audit informatique",
    "62.02B": "Tierce maintenance de systemes et d applications informatiques infogerance assistance technique",
    "62.09Z": "Autres activites informatiques depannage ordinateur installation reseau reparation pc",
    "63.11Z": "Traitement de donnees hebergement et activites connexes cloud hosting serveur",

    # -----------------------------------------------------------------------
    # Section K & L : Finance, Assurance & Immobilier
    # -----------------------------------------------------------------------
    "65.12Z": "Autres assurances assurance dommage iard responsabilite civile multirisque auto habitation",
    "66.22Z": "Activites des agents et courtiers d assurances courtage courtier agent general d assurance",
    "68.10Z": "Activites des marchands de biens immobiliers marchand de biens lotisseur",
    "68.20A": "Location de logements bailleur social hlm oph office public de l habitat residence sociale",
    "68.20B": "Location de terrains et d autres biens immobiliers sci location bureaux commerces entrepots",
    "68.31Z": "Agences immobilieres transaction vente location negociation immobiliere gestion locative mandataire",
    "68.32A": "Administration d immeubles et autres biens immobiliers syndic de copropriete gestionnaire",

    # -----------------------------------------------------------------------
    # Section M : Activités spécialisées, scientifiques et techniques (Professions libérales)
    # -----------------------------------------------------------------------
    "69.10Z": "Activites juridiques avocat cabinet d avocats notaire huissier de justice commissaire priseur conseil juridique",
    "69.20Z": "Activites comptables expert comptable cabinet d expertise comptable commissaire aux comptes tenue de livres",
    "70.22Z": "Conseil pour les affaires et autres conseils de gestion consultant management strategie audit",
    "71.11Z": "Activites d architecture architecte cabinet d architecture maitrise d oeuvre paysagiste",
    "71.12B": "Ingenierie etudes techniques bureau d etudes geometre expert diagnostic immobilier geologie",
    "71.20B": "Controle technique automobile centre auto bilan securite",
    "73.11Z": "Activites des agences de publicite communication marketing agence web creation graphique",
    "74.10Z": "Activites specialisees de design designer decorateur d interieur stylisme graphiste",
    "74.20Z": "Activites photographiques photographe studio photo reportage portrait tirage",
    "75.00Z": "Activites veterinaires clinique veterinaire cabinet veterinaire soins aux animaux medecin veterinaire",

    # -----------------------------------------------------------------------
    # Section N : Activités de services administratifs et de soutien
    # -----------------------------------------------------------------------
    "77.11A": "Location de courte duree de voitures et de vehicules automobiles legers loueur auto",
    "78.10Z": "Activites des agences de placement de main d oeuvre recrutement chasseur de tetes",
    "78.20Z": "Activites des agences de travail temporaire interim agence d interim travail temporaire",
    "79.11Z": "Activites des agences de voyage voyagiste sejour billet d avion",
    "80.10Z": "Activites de securite privee gardiennage surveillance vigile patrouille protection",
    "81.21Z": "Nettoyage courant des batiments entreprise de nettoyage entretien locaux menage bureaux",
    "81.22Z": "Autres activites de nettoyage des batiments et nettoyage industriel lavage vitres desinfection",
    "81.30Z": "Services d amenagement paysager jardinier paysagiste entretien espaces verts elagage tonte creation parcs",

    # -----------------------------------------------------------------------
    # Section P & Q : Enseignement, Santé humaine et action sociale
    # -----------------------------------------------------------------------
    "85.53Z": "Enseignement de la conduite auto ecole permis de conduire moto ecole ecole de conduite",
    "86.10Z": "Activites hospitalieres hopital clinique centre hospitalier soins medicaux hospitalisation urgences",
    "86.21Z": "Activite des medecins generalistes medecine generale docteur en medecine cabinet medical consultation praticien",
    "86.22A": "Activites de radiodiagnostic et de radiotherapie centre d imagerie medicale radiologie echographie irm scanner",
    "86.22B": "Activites chirurgicales chirurgien chirurgie esthetique ambulatoire",
    "86.22C": "Autre activite des medecins specialistes cardiologue dermatologue pediatre ophtalmologue orl gynecologue psychiatre",
    "86.23Z": "Pratique dentaire chirurgien dentiste cabinet dentaire soins dentaires orthodontiste parodontologie implantologie prothese dentaire dentisterie",
    "86.90A": "Ambulances transport sanitaire vehicule de secours vsl ambulancier",
    "86.90B": "Laboratoires d analyses medicales biologie medicale prise de sang prelevement",
    "86.90D": "Activites des infirmiers et des sages femmes soins infirmiers a domicile cabinet infirmier perfusions pansements",
    "86.90E": "Activites des masseurs kinesitherapeutes cabinet de kinesitherapie kine reeducation reeducateur osteopathe",
    "86.90F": "Activites de sante humaine non classees ailleurs pedicurie podologue orthophoniste psychologue dieteticien ergotherapeute chiropracteur",
    "87.10A": "Hebergement medicalise pour personnes agees ehpad maison de retraite medicalisee usld",
    "87.20A": "Hebergement medicalise pour personnes handicapees mas fam foyer d accueil medicalise foyer de vie",
    "88.91A": "Accueil de jeunes enfants creche halte garderie micro creche assistante maternelle garderie periscolaire",
    "88.10A": "Aide a domicile maintien a domicile personnes agees ou handicapees auxiliaire de vie service a la personne admr",

    # -----------------------------------------------------------------------
    # Section R & S : Arts, spectacles, loisirs et autres services personnels
    # -----------------------------------------------------------------------
    "93.11Z": "Gestion d installations sportives gymnase stade piscine court de tennis complexe sportif",
    "93.12Z": "Activites de clubs de sports club de football rugby basketball club sportif association sportive",
    "93.13Z": "Activites des centres de culture physique salle de sport fitness musculation remise en forme crossfit",
    "95.11Z": "Reparation d ordinateurs et d equipements peripheriques depannage informatique atelier pc mac tablette",
    "95.21Z": "Reparation de produits electroniques grand public television hifi audio video",
    "95.23Z": "Reparation de chaussures et d articles en cuir cordonnier cordonnerie serrurerie minute cles",
    "96.01A": "Blanchisserie teinturerie de gros pressing industriel nettoyage a sec",
    "96.01B": "Blanchisserie teinturerie de detail pressing laverie libre service",
    "96.02A": "Coiffure salon de coiffure coiffeur coiffeuse coupe brushing barbe barbier coiffure mixte homme femme enfant a domicile",
    "96.02B": "Soins de beaute institut de beaute esthetique estheticienne soins du visage manucure ongles onglerie epilation massage bien etre spa",
    "96.03Z": "Services funeraires pompes funebres marbrerie funeraire organisation d obseques chambre funeraire",
    "96.09Z": "Autres services personnels toilettage d animaux tatouage percage voyance agence matrimoniale",
}
