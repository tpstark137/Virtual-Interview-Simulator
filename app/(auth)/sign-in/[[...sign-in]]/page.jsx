import { SignIn } from "@clerk/nextjs";

export default function Page() {
  return (
    <section className="bg-white">
      <div className="lg:grid lg:min-h-screen lg:grid-cols-12">
        <section className="relative flex h-32 items-end bg-gray-900 lg:col-span-5 lg:h-full xl:col-span-6">
          <img
            alt=""
            src="https://lh3.googleusercontent.com/p/AF1QipMLGjHfEQ1vYw3xhb4JcJTKMDGZzgYLRsiNHkvT=s1360-w1360-h1020"
            className="absolute inset-0 h-full w-full object-cover opacity-80"
          />

          <div className="hidden lg:relative lg:block lg:p-12">
            <h2 className="mt-6 text-2xl font-extrabold text-white sm:text-3xl md:text-4xl">
              Welcome to Virtual AI Interview
            </h2>

            <p className="mt-4 leading-relaxed text-white/90">
              ECE Major Project
            </p>
          </div>
        </section>

        <main className="flex items-center justify-center px-8 py-8 sm:px-12 lg:col-span-7 lg:px-16 lg:py-12 xl:col-span-6">
          <div className="max-w-xl lg:max-w-3xl">
            <div className="relative -mt-16 block lg:hidden">
              <a
                className="inline-flex size-16 items-center justify-center rounded-full bg-white text-blue-600 sm:size-20"
                href="#"
              >
                <span className="sr-only">Home</span>
              </a>

              <h1 className="mt-2 text-2xl font-bold text-gray-900 sm:text-3xl md:text-4xl">
                Welcome to Virtual AI Interview
              </h1>

              <p className="mt-4 leading-relaxed text-gray-500">
                ECE Major Project
              </p>
            </div>
            <SignIn />
          </div>
        </main>
      </div>
    </section>
  );
}
