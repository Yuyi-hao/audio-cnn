import { Button } from "@/components/ui/button";
import { Home, Search } from "lucide-react";
import { useNavigate } from "react-router-dom"; 

const NotFoundPage = () => {
    const navigate = useNavigate();
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-stone-50 text-center px-4">
      <img
        src="/404cat-imgwebp.webp"
        alt="Confused cat"
        className="w-100 object-contain mb-6"
      />
      <h1 className="text-4xl font-bold text-stone-700 mb-2">404 - Lost in Space</h1>
      <p className="text-stone-500 mb-6 text-lg max-w-xl">
        Uh-oh! Looks like the page you’re looking for wandered off. Maybe try a snack break or head back home?
      </p>
      <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-8">
        <Button onClick={() => navigate("/")} className="bg-emerald-700 hover:bg-emerald-500 text-white w-full sm:w-auto"><Home className="mr-2 size-4"/> Back to Home</Button>
        <Button asChild className="bg-neutral-700 text-white border-neutral-700 w-full sm:w-auto">
            <a href="https://google.com" target="_blank" rel="noopener noreferrer">
                Search on Google <Search className="ml-2" />
            </a>
        </Button>
      </div>
    </div>
  );
};

export default NotFoundPage;
