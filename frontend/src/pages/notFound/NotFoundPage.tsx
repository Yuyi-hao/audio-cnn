import { Button } from "@/components/ui/button";
import { Home, Search } from "lucide-react";
import { useNavigate } from "react-router-dom";

const NotFoundPage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-neutral-900 text-center px-6">
      <img
        src="/404cat-img.webp" // replace with your preferred chill/cute 404 cat
        alt="Chill cat on a 404 sign"
        className="w-64 sm:w-80 object-contain mb-6 drop-shadow-md"
      />

      <h1 className="text-5xl font-extrabold text-stone-200 mb-3">
        404 - Even Zoro Couldn’t Find This Page
      </h1>

      <p className="text-stone-400 text-lg max-w-2xl mb-4">
        This page seems to be hiding better than One Piece. Maybe a cat sat on it, or Zoro read the map upside down again.
      </p>

      <p className="text-stone-500 text-base max-w-xl mb-8">
        Either way, it’s not here. Try heading back home or let Google take a shot at it — they’ve got better direction than Zoro.
      </p>

      <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
        <Button
          onClick={() => navigate("/")}
          className="bg-emerald-600 hover:bg-emerald-500 text-white w-full sm:w-auto"
        >
          <Home className="mr-2 size-4" />
          Back to Home
        </Button>

        <Button
          asChild
          className="bg-neutral-700 text-white border-neutral-700 w-full sm:w-auto"
        >
          <a href="https://google.com" target="_blank" rel="noopener noreferrer">
            Search on Google <Search className="ml-2" />
          </a>
        </Button>
      </div>

      <img
        src="/zorolost.gif" // new zoro gif
        alt="Zoro wandering around confused"
        className="mt-12 w-40 sm:w-52 object-contain opacity-80"
      />
    </div>
  );
};

export default NotFoundPage;
