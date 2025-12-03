import { useState } from "react";
import { Search, Fish, Info, Ruler } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface Species {
  name: string;
  scientificName: string;
  habitat: string;
  legalSize: string;
  description: string;
  characteristics: string[];
  image: string;
  region: string;
}

const indiaSpeciesData: Species[] = [
  {
    name: "Rohu",
    scientificName: "Labeo rohita",
    habitat: "Freshwater rivers and lakes",
    legalSize: "30 cm minimum",
    description: "Rohu is one of the most important freshwater fish in India, widely cultivated and consumed.",
    characteristics: ["Silver body", "Large scales", "Pointed snout", "Omnivorous feeder"],
    image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNDA5MGU1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPlJvaHU8L3RleHQ+PC9zdmc+",
    region: "Northern & Eastern India"
  },
  {
    name: "Catla",
    scientificName: "Catla catla",
    habitat: "Large rivers and reservoirs",
    legalSize: "40 cm minimum",
    description: "Catla is the fastest-growing Indian major carp and an important aquaculture species.",
    characteristics: ["Large head", "Upturned mouth", "Silver-golden color", "Surface feeder"],
    image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNzFhNjg1Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkNhdGxhPC90ZXh0Pjwvc3ZnPg==",
    region: "Across India"
  },
  {
    name: "Hilsa",
    scientificName: "Tenualosa ilisha",
    habitat: "Marine and freshwater (anadromous)",
    legalSize: "23 cm minimum",
    description: "Hilsa is the national fish of Bangladesh and highly prized in Bengali cuisine.",
    characteristics: ["Silvery body", "Compressed laterally", "Small scales", "Seasonal migration"],
    image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjM5YzEyIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkhpbHNhPC90ZXh0Pjwvc3ZnPg==",
    region: "Eastern India, West Bengal"
  },
  {
    name: "Pomfret",
    scientificName: "Pampus argenteus",
    habitat: "Coastal marine waters",
    legalSize: "15 cm minimum",
    description: "Silver pomfret is a highly valued marine fish popular in Indian coastal cuisine.",
    characteristics: ["Flat, disc-shaped body", "Silver coloration", "No pelvic fins", "Delicate flesh"],
    image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjODQ3NDcwIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPlBvbWZyZXQ8L3RleHQ+PC9zdmc+",
    region: "All Indian coasts"
  },
  {
    name: "Kingfish",
    scientificName: "Scomberomorus commerson",
    habitat: "Open ocean and coastal waters",
    legalSize: "50 cm minimum",
    description: "King mackerel is a popular game fish and important commercial species in Indian waters.",
    characteristics: ["Streamlined body", "Sharp teeth", "Metallic blue-green back", "Fast swimmer"],
    image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNWY2MzY4Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iS2luZ2Zpc2g8L3RleHQ+PC9zdmc+",
    region: "Indian Ocean coasts"
  },
  {
    name: "Mrigal",
    scientificName: "Cirrhinus mrigala",
    habitat: "Rivers and ponds",
    legalSize: "25 cm minimum",
    description: "Mrigal is one of the three Indian major carps, important for aquaculture.",
    characteristics: ["Olive green back", "Lateral line scales", "Bottom feeder", "Hardy species"],
    image: "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjNmY4ZjRlIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNiIgZmlsbD0iI2ZmZiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPk1yaWdhbDwvdGV4dD48L3N2Zz4=",
    region: "Northern India"
  }
];

export const SpeciesLibrary = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedSpecies, setSelectedSpecies] = useState<Species | null>(null);

  const filteredSpecies = indiaSpeciesData.filter(species =>
    species.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    species.scientificName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    species.region.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold flex items-center justify-center gap-2 mb-2">
          <Fish className="w-6 h-6" />
          Species Reference
        </h1>
        <p className="text-muted-foreground">Offline guide to Indian fish species</p>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          placeholder="Search species, scientific name, or region..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* Species Grid */}
      <div className="grid grid-cols-1 gap-4">
        {filteredSpecies.map((species) => (
          <Card 
            key={species.name}
            className="cursor-pointer hover:shadow-ocean hover:border-primary/20 transition-all"
            onClick={() => setSelectedSpecies(species)}
          >
            <CardContent className="p-4">
              <div className="flex gap-4">
                <img
                  src={species.image}
                  alt={species.name}
                  className="w-20 h-16 object-cover rounded-lg bg-muted flex-shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-semibold text-foreground">{species.name}</h3>
                      <p className="text-sm text-muted-foreground italic">{species.scientificName}</p>
                    </div>
                    <Badge variant="outline" className="text-xs">
                      {species.region}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Ruler className="w-3 h-3" />
                    {species.legalSize}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Species Detail Dialog */}
      <Dialog open={!!selectedSpecies} onOpenChange={(open) => !open && setSelectedSpecies(null)}>
        <DialogContent className="max-w-2xl">
          {selectedSpecies && (
            <div className="space-y-6">
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Fish className="w-5 h-5" />
                  {selectedSpecies.name}
                </DialogTitle>
              </DialogHeader>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <img
                    src={selectedSpecies.image}
                    alt={selectedSpecies.name}
                    className="w-full h-48 object-cover rounded-lg bg-muted"
                  />
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Scientific Name:</span>
                      <span className="text-sm italic">{selectedSpecies.scientificName}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Legal Size:</span>
                      <Badge variant="secondary">{selectedSpecies.legalSize}</Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Region:</span>
                      <span className="text-sm">{selectedSpecies.region}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2 flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      Description
                    </h4>
                    <p className="text-sm text-muted-foreground">{selectedSpecies.description}</p>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Habitat</h4>
                    <p className="text-sm text-muted-foreground">{selectedSpecies.habitat}</p>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Key Characteristics</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedSpecies.characteristics.map((char) => (
                        <Badge key={char} variant="outline" className="text-xs">
                          {char}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};