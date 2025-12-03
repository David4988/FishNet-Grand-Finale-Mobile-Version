import { SpeciesLibrary } from '@/components/reference/SpeciesLibrary';

export default function SpeciesPage() {
  return (
    <div className="min-h-screen bg-background p-4">
      <div className="container mx-auto max-w-md">
        <SpeciesLibrary />
      </div>
    </div>
  );
}